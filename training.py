import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import json
from dataset import profiles_dataset
from NN import MLP, bVAE, CNN
import time, os, glob
from torchsummary import summary
import wandb

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

# Parse configuration file
config_file = 'config.json'
with open(config_file) as json_file:
    hyperparameters = json.load(json_file)

timestr = time.strftime("%Y%m%d-%H%M%S")
hyperparameters['timestr'] = timestr

# Definition parameters of the network
coefficient = hyperparameters['coefficient']
archiquecture = hyperparameters['archiquecture']
readir = hyperparameters['readir']
gpu = hyperparameters['gpu']

if archiquecture == 'mlp':
    print('Using MLP')
    mlp_hidden_size = hyperparameters['params'][archiquecture]['mlp_hidden_size']
elif archiquecture == 'cnn':
    print('Using CNN')
    cnn_hidden_size = hyperparameters['params'][archiquecture]['cnn_hidden_size']
    mlp_hiden_in = hyperparameters['params'][archiquecture]['mlp_hiden_in']
    mlp_hiden_out = hyperparameters['params'][archiquecture]['mlp_hiden_out']
    conv_kernel_size = hyperparameters['params'][archiquecture]['conv_kernel_size']
elif archiquecture == 'bvae':
    print('Using bVAE')
    bvae_enc_size = hyperparameters['params'][archiquecture]['bvae_enc_size']
    bvae_dec_size = hyperparameters['params'][archiquecture]['bvae_dec_size']
    bvae_latent_size = hyperparameters['params'][archiquecture]['bvae_latent_size']
    bvae_beta = hyperparameters['params'][archiquecture]['bvae_beta']
else:
    raise ValueError(f'The architecture "{archiquecture}" is not defined')

# Training network parameters
batch_size = hyperparameters['batch_size']
epochs = hyperparameters['epochs']
learning_rate = hyperparameters['learning_rate']
step_size_scheduler = hyperparameters['step_size_scheduler']
gamma_scheduler = hyperparameters['gamma_scheduler']

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
# construct the base name to save the model
basename = f'trained_model_{archiquecture}_{coefficient}_{hyperparameters["group_suffix"]}'
savedir = f'./checkpoints/{basename}_time_{timestr}/'
# check if there is a folder for the checkpoints and create it if not
if not os.path.exists(savedir):
    os.makedirs(savedir)
# file to load the data from
readfile = f'model_ready_{coefficient}_{hyperparameters["dataset"]}.pkl'
hyperparameters['dataset'] = readfile
print('Reading data from: ', readir + readfile)

wandb.init(project="neural-He", name=f"{archiquecture}-{coefficient}-{timestr}", entity="solar-iac",
           group = f"{archiquecture}-{hyperparameters['group_suffix']}-{hyperparameters['dataset']}", job_type = f"{coefficient}",
           config=hyperparameters, save_code=True, magic=True)

# check if the GPU is available
cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
print('Checking the GPU availability...')
if cuda:
    print('GPU is available')
    print('Using GPU {}'.format(gpu))
else:
    print('GPU is not available')
    print('Using CPU')
    print(device)

# If the nvidia_smi package is installed, then report some additional information
if (NVIDIA_SMI):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
    print(f"Computing in {device} : {nvidia_smi.nvmlDeviceGetName(handle)}")

# create the training and test dataset
print('-'*50)
print('Creating the training dataset ...')
dataset = profiles_dataset(f'{readir}{readfile}', train=True, archiquecture=archiquecture)
print('\nCreating the test dataset ...')
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False, archiquecture=archiquecture)

samples_test = set(test_dataset.indices)
samples_train = set(dataset.indices)

# check that the training and test sets are disjoint
print('\nNumber of training samples: {}'.format(len(samples_train)))
print('Number of test samples: {}'.format(len(samples_test)))
print('Number of samples in both sets: {}'.format(len(samples_train.intersection(samples_test))))
assert(len(samples_test.intersection(samples_train)) == 0)
print('Training and test sets are disjoint!\n')

# DataLoader is used to load the dataset for training and testing
print('Creating the training DataLoader ...')
train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           pin_memory = False,
                                           num_workers = 4)
print('Creating the test DataLoader ...')
# use only 4 cpus for loading
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          pin_memory = False,
                                          num_workers = 4)

# Model Initialization
print('-'*50)
print('Initializing the model ...\n')

if archiquecture == 'mlp':
    model = MLP(dataset.n_features, dataset.n_components, mlp_hidden_size).to(device)
elif archiquecture == 'cnn':
    model = CNN(dataset.n_components, dataset.n_features, mlp_hiden_in, 
                mlp_hiden_out, cnn_hidden_size, conv_kernel_size).to(device)
elif archiquecture == 'bvae':
    model = bVAE(dataset.n_components, dataset.n_features, bvae_latent_size,
                 bvae_enc_size, bvae_dec_size, bvae_beta).to(device)
else:
    raise ValueError('The architecture is not defined')


# print the dataset dimensions
print('Dataset dimensions:')
print('Number of components: {}'.format(dataset.n_components))
print('Number of features: {}'.format(dataset.n_features))
print('Number of batches: {}'.format(len(train_loader)))
print('Batch size: {}\n'.format(batch_size))

summary(model, (1, dataset.n_features), batch_size=batch_size)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size_scheduler,
                                            gamma=gamma_scheduler)

train_losses = []
test_losses = []
best_loss = float('inf')

# start training
print('-'*50)
print('Training the model ...\n')
start_time = time.time()
print('Start time: {}'.format(time.ctime()))
print('-'*50 + '\n')
for epoch in range(epochs):
    train_loss_epoch = 0
    model.train()
    for (params, profiles) in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}", leave = False):

        params = params.to(device)
        profiles = profiles.to(device)

        # Forward pass
        reconstructed = model(params)

        # Calculating the loss function
        train_loss = loss_function(reconstructed, profiles)

        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Compute loss
        train_loss_epoch += train_loss.item()

    # Storing the losses in a list for plotting
    train_loss_epoch /= len(train_loader)
    train_losses.append(train_loss_epoch)

    test_loss_epoch = 0
    model.eval()
    with torch.no_grad():
        for (params, profiles) in tqdm(test_loader, desc = f"Epoch Validation {epoch}/{epochs}", leave = False):

            params = params.to(device)
            profiles = profiles.to(device)

            # Forward pass
            reconstructed = model(params)
            # Calculating the loss function
            test_loss = loss_function(reconstructed, profiles)
            # Compute test loss
            test_loss_epoch += test_loss.item()

    # Storing the losses in a list for plotting
    test_loss_epoch /= len(test_loader)
    test_losses.append(test_loss_epoch)

    print(f"Epoch {epoch}: Train Loss: {train_loss_epoch:1.2e},"+
                        f" Test Loss: {test_loss_epoch:1.2e},"+
                        f" lr: {scheduler.get_last_lr()[0]:1.2e}")

    if (test_loss_epoch < best_loss):
        best_loss = test_loss_epoch

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_loss': train_loss_epoch,
            'valid_loss': test_loss_epoch,
            'best_loss': best_loss,
            'hyperparameters': hyperparameters,
            'optimizer': optimizer.state_dict()}

        print("Saving best model with hyperparameters...")
        filename = f'{basename}_{time.strftime("%Y%m%d-%H%M%S")}'
        torch.save(checkpoint, f'{savedir}' + filename + '.pth')

    scheduler.step()
    wandb.log({
                'train_loss': train_loss_epoch,
                'valid_loss': test_loss_epoch,
                'learning_rate': scheduler.get_last_lr()[0],
                'best_loss': best_loss,
              })
    # Optional
    wandb.watch(model)

# finished training
end_time = time.time()
print('End time: {}'.format(time.ctime()))
print('Training time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
print('-'*50)

print('Saving the model ...')
filename = f'{basename}_end_{time.strftime("%Y%m%d-%H%M%S")}'
torch.save(checkpoint, f'{savedir}' + filename + '.pth')
print('Model saved!\n')
print('-'*50)

# saving the losses
print('Saving the losses ...')
filename = f'{basename}_losses_{time.strftime("%Y%m%d-%H%M%S")}'
np.savez(f'{savedir}' + filename + '.npz', train_losses, test_losses)
print('Losses saved!\n')
