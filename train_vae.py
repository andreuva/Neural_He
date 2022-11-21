import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
from modules import siren
import argparse
from modules import vae
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import h5py
import pathlib

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_training=None):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        root = '/scratch1/aasensio/3dcubes'

        files = ['rempel_modelB.h5', 'cheung_model.h5', 'cheung_model2B.h5']
        # files = ['cheung_model2B.h5']

        self.tau = np.linspace(-1.0, 1.0, 51)

        models = []

        for i, f in enumerate(files):
            print(f'Reading {f}')
            f = h5py.File(f'{root}/{f}', 'r')

            nx, ny, nq, nz = f['model'].shape
            
            models.append(f['model'][:].reshape((nx*ny, nq, nz)))

        self.model = np.concatenate(models, axis=0)

        self.model = np.transpose(self.model, axes=(0,2,1))

        # dz, T, P, vz, Bx, By, Bz
        # Select only T, vz, Bx, By, Bz
        ind = [1, 3, 4, 5, 6]
        self.model = self.model[:, :, ind]

        # Now rescale
        self.model[..., 0] = self.model[..., 0] / 8000.0
        self.model[..., 1] = self.model[..., 1] / 1e6
        self.model[..., 2] = self.model[..., 2] / 2e3
        self.model[..., 3] = self.model[..., 3] / 2e3
        self.model[..., 4] = self.model[..., 4] / 2e3

        for i in range(5):
            print(f'{i} -> min={np.min(self.model[..., i])} - max = {np.max(self.model[..., i])} - std={np.std(self.model[..., i])}')

        if (n_training is not None):
            self.model = self.model[0:n_training, ...]

        self.n_training = self.model.shape[0]

        idx = np.arange(self.n_training)
        np.random.shuffle(idx)

        self.model = self.model[idx, ...]
        
    def __getitem__(self, index):

        return self.tau.astype('float32'), self.model[index, :, :].astype('float32'), index.astype('long')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Training(object):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        
        self.cuda = torch.cuda.is_available()
        self.gpu = self.hyperparameters['gpu']
        self.smooth = self.hyperparameters['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        self.validation_split = self.hyperparameters['validation_split']
        self.latent_size = self.hyperparameters['latent_size']

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = self.hyperparameters['batchsize']
        self.validation_split = self.hyperparameters['validation_split']
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}        
                
        self.dataset = Dataset(n_training=None)
        
        idx = np.arange(self.dataset.n_training)
        
        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

        # Model
        self.model = vae.BetaVAE(dim_in = len(self.dataset.tau), 
                dim_out = self.hyperparameters['n_out'],
                enc_n_hidden = self.hyperparameters['enc_n_hidden'], 
                enc_dim_hidden = self.hyperparameters['enc_dim_hidden'], 
                latent_size = self.hyperparameters['latent_size'], 
                siren_n_hidden = self.hyperparameters['siren_n_hidden'],
                siren_dim_hidden = self.hyperparameters['siren_dim_hidden'], 
                siren_w0 = self.hyperparameters['w0'], 
                film_n_hidden = self.hyperparameters['film_n_hidden'], 
                film_dim_hidden = self.hyperparameters['film_dim_hidden'],
                beta = self.hyperparameters['beta']).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))        

    def init_optimize(self, epochs, lr, weight_decay, scheduler):

        self.lr = self.hyperparameters['lr']
        self.weight_decay = self.hyperparameters['wd']
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = self.hyperparameters['n_epochs']
        
        p = pathlib.Path('trained_vae/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'trained_vae/{0}'.format(current_time)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.1*lr)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss_avg = self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

            if (loss_avg < best_loss):
                print(f"Saving model {self.out_name}.pth")
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'hyperparameters': self.hyperparameters
                }
                
                best_loss = loss_avg
                torch.save(checkpoint, f'{self.out_name}.pth')
        
    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (inputs, outputs, indices) in enumerate(t):
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            indices = indices.to(self.device)

            self.optimizer.zero_grad()
            
            recons, z, mu, logvar = self.model.forward(outputs, inputs[:, :, None])
            
            loss, recons_loss, kld_loss = self.model.loss_function(recons, outputs, mu, logvar)
                                
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
                recons_loss_avg = recons_loss.item()
                kld_loss_avg = kld_loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                recons_loss_avg = self.smooth * recons_loss.item() + (1.0 - self.smooth) * recons_loss_avg
                kld_loss_avg = self.smooth * kld_loss.item() + (1.0 - self.smooth) * kld_loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                t.set_postfix(loss=loss_avg, loss_l2=recons_loss_avg, loss_kl=kld_loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)                
            else:
                t.set_postfix(loss=loss_avg, loss_l2=recons_loss_avg, loss_kl=kld_loss_avg, lr=current_lr)
            
        self.loss.append(loss_avg)

        device = 'cpu' if self.device.type == 'cpu' else f'cuda:{self.device.index}'
        
        return loss_avg

    def test(self, epoch):
        self.model.eval()
        t = tqdm(self.validation_loader)
        n = 1
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, outputs, indices) in enumerate(t):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                indices = indices.to(self.device)

                recons, z, mu, logvar = self.model.forward(outputs, inputs[:, :, None])
            
                loss, recons_loss, kld_loss = self.model.loss_function(recons, outputs, mu, logvar)
                                        
                if (batch_idx == 0):
                    loss_avg = loss.item()
                    recons_loss_avg = recons_loss.item()
                    kld_loss_avg = kld_loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                    recons_loss_avg = self.smooth * recons_loss.item() + (1.0 - self.smooth) * recons_loss_avg
                    kld_loss_avg = self.smooth * kld_loss.item() + (1.0 - self.smooth) * kld_loss_avg
                
                t.set_postfix(loss=loss_avg, loss_l2=recons_loss_avg, loss_kl=kld_loss_avg)
            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):

    hyperparameters = {
        'batchsize': 4096,
        'validation_split': 0.2,
        'gpu': 0,
        'smooth': 0.15,
        'enc_n_hidden': 5,
        'enc_dim_hidden': 128,
        'latent_size': 32,
        'siren_n_hidden': 5,
        'siren_dim_hidden': 128,
        'w0': 10.0,
        'film_n_hidden': 5,
        'film_dim_hidden': 128,
        'n_out': 5,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 200,
        'beta': 1e-6
    }

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')    
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.15, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--scheduler', '--scheduler', default=10, type=int,
                    metavar='SCHEDULER', help='Number of epochs before applying scheduler')
    parser.add_argument('--batch', '--batch', default=64, type=int,
                    metavar='BATCH', help='Batch size')
    
    parsed = vars(parser.parse_args())

    deepnet = Training(hyperparameters)

    deepnet.init_optimize(parsed['epochs'], lr=parsed['lr'], weight_decay=parsed['wd'], scheduler=parsed['scheduler'])
    deepnet.optimize()