from pyexpat import features
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import time
from configobj import ConfigObj
from datasets import spectral_dataset
from NN import EncoderDecoder

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


# Define the model class to store the model and all the methods to train and test the model
class Model(object):
    def __init__(self, config, gpu):
    
        # check if the GPU is available
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        if self.cuda:
            print('GPU is available')
            print('Using GPU {}'.format(self.gpu))
        else:
            print('GPU is not available')
            print('Using CPU')
            print(self.device)

        # If the nvidia_smi package is installed, then report some additional information
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print(f"Computing in {self.device} : {nvidia_smi.nvmlDeviceGetName(self.handle)}")

        # Parse configuration file and transform to integers
        self.hyperparameters = ConfigObj(infile=config)
        self.encoder_dimensions = [int(x) for x in self.hyperparameters['encoder_dimensions']]
        self.decoder_dimensions = [int(x) for x in self.hyperparameters['decoder_dimensions']]

        print('\nEncoder dimensions: {0}'.format(self.encoder_dimensions))
        print('Decoder dimensions: {0}'.format(self.decoder_dimensions))

        # Define the model
        self.model = EncoderDecoder(self.encoder_dimensions,self.decoder_dimensions).to(self.device)
        # Print the number of trainable parameters
        print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def train(self, epochs, lr, batch_size, split, smooth, readir, savedir):

        self.train_split = split
        self.lr = lr
        self.smooth = smooth
        self.n_epochs = epochs
        self.batch_size = batch_size
        
        # create the dataset
        self.dataset = spectral_dataset(readir)

        # Randomly shuffle a vector with the indices to separate between training/validation datasets
        idx = np.arange(self.dataset.n_samples)
        np.random.shuffle(idx)

        self.train_index = idx[0:int(self.train_split*self.dataset.n_samples)]
        self.validation_index = idx[int(self.train_split*self.dataset.n_samples):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        # Define the data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(
            self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False)

        # Define the name of the model
        filename = f'{time.strftime("%Y%m%d-%H%M%S")}_modelparams.dat'
        print('saving model hyperparams at Model: {0}'.format(savedir + filename))
        filepath = savedir + filename
        self.hyperparameters.filename = filepath
        self.hyperparameters.write()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

        # Loss function
        self.loss_fn = torch.nn.MSELoss()

        # Now start the training
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):

            # Compute training and validation steps
            train_loss = self.optimize(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # If the validation loss improves, save the model as best
            if (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': self.hyperparameters,
                    'optimizer': self.optimizer.state_dict()}

                print("Saving best model...")
                filename = f'checkpoint_{time.strftime("%Y%m%d-%H%M%S")}.pth'
                torch.save(checkpoint, savedir + filename + '_best.pth')

            # Update the learning rate
            self.scheduler.step()

    def optimize(self, epoch):

        # Put the model in training mode
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0

        for batch_idx, (features, target) in enumerate(t):

            # Move them to the GPU
            features = features.to(self.device)
            target = target.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Evaluate the model
            out = self.model(features)

            # Compute loss
            loss = self.loss_fn(out.squeeze(), target.squeeze())

            # Compute backpropagation
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            # Compute smoothed loss
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update information for this batch
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu,
                              memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        # Do a validation of the model and return the loss
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(t):

                # Move them to the GPU
                features = features.to(self.device)
                target = target.to(self.device)

                out = self.model(features)

                loss = self.loss_fn(out.squeeze(), target.squeeze())

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                t.set_postfix(loss=loss_avg)

        return loss_avg

    def summary(self):
        # # Plotting the last 100 values
        plt.plot([los for los in self.train_loss], label='training')
        plt.plot([los for los in self.valid_loss], label='validation')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        # select a random sample from the test dataset and test the network
        # then plot the predicted output and the ground truth
        self.model.eval()

        self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index[np.random.randint(0, len(self.validation_index), 5)])
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, sampler=self.test_sampler, batch_size=1, shuffle=False)
        t = tqdm(self.test_loader)
        loss_avg = 0
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(t):

                # Move them to the GPU
                features = features.to(self.device)
                target = target.to(self.device)

                out = self.model(features)

                # Compute loss
                loss = self.loss_fn(out.squeeze(), target.squeeze())
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                t.set_postfix(loss=loss_avg)

                # bring back the output to the CPU
                out = out.squeeze().cpu()
                features = features.squeeze().cpu()

                fft_rec = out.detach().numpy()
                fft_reconst = np.zeros(self.dataset.n_components, dtype=np.complex64)
                fft_reconst.real = fft_rec[:self.dataset.n_components]
                fft_reconst.imag = fft_rec[self.dataset.n_components:]

                reconstructed = np.fft.irfft(fft_reconst, n=self.dataset.n_features)
                plt.plot(features, 'or')
                plt.plot(reconstructed, 'b')
            
        print(f'Loss in testing: {loss_avg}')
        # show the plot
        plt.show()
