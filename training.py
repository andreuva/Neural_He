import numpy as np
from tqdm import tqdm
import torch
import json
from dataset import profiles_dataset
from NN import bVAE
import time, os, glob
# from torchsummary import summary
import wandb

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

# Parse configuration file
config_file = 'config_vae.json'
with open(config_file) as json_file:
    hyperparameters = json.load(json_file)

timestr = time.strftime("%Y%m%d-%H%M%S")
hyperparameters['timestr'] = timestr

# Definition parameters of the network
coefficient = hyperparameters['coefficient']
archiquecture = hyperparameters['archiquecture']
readir = hyperparameters['readir']
gpu = hyperparameters['gpu']

print(f'Using {archiquecture}')
print('Loading the bvae parameters ...')
enc_size = hyperparameters['params']['bvae']['enc_size']
dec_size = hyperparameters['params']['bvae']['dec_size']
latent_size = hyperparameters['params']['bvae']['latent_size']
beta = hyperparameters['params']['bvae']['beta']

# Training network parameters
print('Loading the optimization parameters ...')
batch_size = hyperparameters['batch_size']
epochs = hyperparameters['epochs']
learning_rate = hyperparameters['learning_rate']
step_size_scheduler = hyperparameters['step_size_scheduler']
gamma_scheduler = hyperparameters['gamma_scheduler']

# Create the saving directory
print('Creating the saving directory ...')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
# construct the base name to save the model
basename = f'trained_model_{coefficient}_{hyperparameters["group_suffix"]}_{beta}'
savedir = f'./checkpoints/{basename}_time_{timestr}/'
# check if there is a folder for the checkpoints and create it if not
if not os.path.exists(savedir):
    os.makedirs(savedir)
# file to load the data from
readfile = f'model_ready_{coefficient}_{hyperparameters["dataset"]}.pkl'
hyperparameters['dataset_file'] = readfile
hyperparameters['dataset_dir'] = readir
print('Reading data from: ', readir + readfile)

wandb.init(project="neural-He", name=f"{coefficient}-{beta}-{timestr}", entity="solar-iac",
           group = f"{hyperparameters['group_suffix']}-{hyperparameters['dataset']}-{beta}", job_type = f"{coefficient}",
           save_code=True)

# add the hyperparameters to the wandb run one by one
for key, value in hyperparameters.items():
    wandb.config[key] = value

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
dataset = profiles_dataset(f'{readir}{readfile}', train=True)
print('\nCreating the test dataset ...')
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)

samples_test = set(test_dataset.indices)
samples_train = set(dataset.indices)

# check that the training and test sets are disjoint
print('\nNumber of training samples: {}'.format(len(samples_train)))
print('Number of test samples: {}'.format(len(samples_test)))
print('Number of samples in both sets: {}'.format(len(samples_train.intersection(samples_test))))
assert(len(samples_test.intersection(samples_train)) == 0)
print('Training and test sets are disjoint!\n')

# DataLoader is used to load the dataset for training and testing
# use only 4 cpus for loading the data
print('Creating the training DataLoader ...')
train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, pin_memory = False, num_workers = 4)
print('Creating the test DataLoader ...')
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True, pin_memory = False, num_workers = 4)

# print the dataset dimensions
print('Dataset dimensions:')
print('Number of components: {}'.format(dataset.n_components))
print('Number of features: {}'.format(dataset.n_features))
print('Number of batches: {}'.format(len(train_loader)))
print('Batch size: {}\n'.format(batch_size))

# Model Initialization
print('-'*50)
print('Initializing the model ...\n')
model = bVAE(dataset.n_components, dataset.n_features, latent_size, enc_size, dec_size, beta).to(device)

# summary(model, (1, dataset.n_features), batch_size=batch_size)

# Using an Adam Optimizer with learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_scheduler, gamma=gamma_scheduler)

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
    train_recons_loss_epoch = 0
    train_kld_loss_epoch = 0
    model.train()
    # training loop over the batches in the training set (tqdm for progress bar with 50 columns)
    for (data, labels, params) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100):

        # move the data to the GPU
        data = data.to(device)
        labels = labels.to(device)
        params = params.to(device)

        # Forward pass
        reconstructed, mu, logvar = model(data)
        # Calculating the loss function
        train_loss, train_recons_loss, train_kld_loss  = model.loss_function(reconstructed, labels, mu, logvar)

        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # save the loss for the epoch
        train_loss_epoch += train_loss.item()
        train_recons_loss_epoch += train_recons_loss.item()
        train_kld_loss_epoch += train_kld_loss.item()

    # Storing the losses in a list for plotting
    train_loss_epoch /= len(train_loader)
    train_recons_loss_epoch /= len(train_loader)
    train_kld_loss_epoch /= len(train_loader)
    train_losses.append([train_loss_epoch, train_recons_loss_epoch, train_kld_loss_epoch])

    test_loss_epoch = 0
    test_recons_loss_epoch = 0
    test_kld_loss_epoch = 0
    model.eval()
    with torch.no_grad():
        for (data, labels, params) in tqdm(test_loader, desc = f"Epoch Validation {epoch}/{epochs}", ncols=100):

            data = data.to(device)
            labels = labels.to(device)
            params = params.to(device)

            # Forward pass
            reconstructed, mu, logvar = model(data)
            # Calculating the loss function
            test_loss, test_recons_loss, test_kld_loss  = model.loss_function(reconstructed, labels, mu, logvar)

            # Compute test loss
            test_loss_epoch += test_loss.item()
            test_recons_loss_epoch += test_recons_loss.item()
            test_kld_loss_epoch += test_kld_loss.item()

    # Storing the losses in a list for plotting
    test_loss_epoch /= len(test_loader)
    test_recons_loss_epoch /= len(test_loader)
    test_kld_loss_epoch /= len(test_loader)
    test_losses.append([test_loss_epoch, test_recons_loss_epoch, test_kld_loss_epoch])

    print(f"Epoch {epoch}: Train Loss: {train_loss_epoch:1.2e},"+
                        f" Test Loss: {test_loss_epoch:1.2e},"+
                        f" lr: {scheduler.get_last_lr()[0]:1.2e}")

    if (test_loss_epoch < best_loss):
        best_loss = test_loss_epoch

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_loss': train_loss_epoch,
            'train_recons_loss': train_recons_loss_epoch,
            'train_kld_loss': train_kld_loss_epoch,
            'valid_loss': test_loss_epoch,
            'valid_recons_loss': test_recons_loss_epoch,
            'valid_kld_loss': test_kld_loss_epoch,
            'best_loss': best_loss,
            'hyperparameters': hyperparameters,
            'optimizer': optimizer.state_dict()}

        print("Saving best model with hyperparameters...")
        filename = f'{basename}_{time.strftime("%Y%m%d-%H%M%S")}'
        torch.save(checkpoint, f'{savedir}' + filename + '.pth')

    scheduler.step()
    wandb.log({
                'train_loss': train_loss_epoch,
                'train_recons_loss': train_recons_loss_epoch,
                'train_kld_loss': train_kld_loss_epoch,
                'valid_loss': test_loss_epoch,
                'valid_recons_loss': test_recons_loss_epoch,
                'valid_kld_loss': test_kld_loss_epoch,
                'learning_rate': scheduler.get_last_lr()[0],
                'best_loss': best_loss,
              })
    # Optional
    if (epoch % 25 == 0):
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

wandb.finish()
