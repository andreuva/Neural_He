import numpy as np
from tqdm import tqdm
import torch
import json
from dataset import profiles_dataset, print_hyperparameters
from NN import bVAE, mapping
import time, os, glob
from torchsummary import summary
import wandb

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

############################################################################
#                    Load and configure parameters                         #
############################################################################

# Parse configuration file
config_file = 'config_map.json'
with open(config_file) as json_file:
    hyperparameters_map = json.load(json_file)

timestr = time.strftime("%Y%m%d-%H%M%S")
hyperparameters_map['timestr'] = timestr

# Load the configuration file to get the VAE and the mapping hyperparameters
print('Loading the maping and optimization parameters ...')
hidden_size = hyperparameters_map['hidden_size']

batch_size = hyperparameters_map['batch_size']
epochs = hyperparameters_map['epochs']
learning_rate = hyperparameters_map['learning_rate']
step_size_scheduler = hyperparameters_map['step_size_scheduler']
gamma_scheduler = hyperparameters_map['gamma_scheduler']
gpu = hyperparameters_map['gpu']


# get the last checkpoint
check_folder = hyperparameters_map['readir'] + hyperparameters_map['vae'] + '/'
checkpoint = sorted(glob.glob(f'{check_folder}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {check_folder}')
print(f'Loading the checkpoint {checkpoint[len(check_folder):]}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

# get the hyperparameters from the VAE checkpoint
print('Loading the bvae hyperparameters from the checkpoint ...')
coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = checkpoint['hyperparameters']['dataset_dir']
readfile = checkpoint["hyperparameters"]["dataset_file"]
hyperparameters_vae = checkpoint['hyperparameters']

# construct the base name to save the checkpoints
basename = f'map_trained_model_{hyperparameters_map["group_suffix"]}_{timestr}'
savedir = check_folder + basename + '/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

print('Loading the bvae parameters ...')
enc_size = hyperparameters_vae['params']['bvae']['enc_size']
dec_size = hyperparameters_vae['params']['bvae']['dec_size']
latent_size = hyperparameters_vae['params']['bvae']['latent_size']
beta = hyperparameters_vae['params']['bvae']['beta']

# print the hyperparameters (tabulated correctly) and loss
print('\nHyperparameters of the VAE')
print('-'*100)
print_hyperparameters(hyperparameters_vae)
print(f'{"checkpoint training loss vae":<25}: {checkpoint["train_loss"]}')
print(f'{"checkpoint validation loss vae":<25}: {checkpoint["valid_loss"]}')
print('-'*100 + '\n')

# add the vae hyperparameters to the map hyperparameters
hyperparameters_map['vae_hyperparameters'] = hyperparameters_vae

############################################################################
#                           GPU CONFIGURATION                              #
############################################################################

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

############################################################################
#                    Reading data and load model                           #
############################################################################

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)
train_dataset = profiles_dataset(f'{readir}{readfile}', train=True)

# load the vae model
print('Loading the vae model ...')
vae = bVAE(test_dataset.n_components, test_dataset.n_features, latent_size, enc_size, dec_size, beta).to(device)
vae.load_state_dict(checkpoint['state_dict'])

# set gradient to false
print('Setting the gradient to false ...')
for param in vae.parameters():
    param.requires_grad = False


wandb.init(project="neural-He", name=f"map-{coefficient}-{timestr}", entity="solar-iac",
           group = f"map-{hyperparameters_map['group_suffix']}-{hyperparameters_vae['dataset']}", job_type = f"{coefficient}",
           save_code=True)

# add the hyperparameters to the wandb run one by one
for key, value in hyperparameters_map.items():
    wandb.config[key] = value


# create the training and test dataset
print('-'*100)
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
print('-'*100)
print('Initializing the maping model ...\n')
model = mapping(dataset.n_params, latent_size, hidden_size).to(device)

summary(model, (1, dataset.n_params), batch_size=batch_size)

# Using an Adam Optimizer with learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_scheduler, gamma=gamma_scheduler)

train_losses = []
test_losses = []
best_loss = float('inf')

############################################################################
#                        Training of the model                             #
############################################################################
# start training
print('-'*100)
print('Training the model ...\n')
start_time = time.time()
print('Start time: {}'.format(time.ctime()))
print('-'*100 + '\n')
for epoch in range(epochs):
    train_loss_epoch = 0
    model.train()
    # training loop over the batches in the training set (tqdm for progress bar with 50 columns)
    for (data, labels, params) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100):

        # move the data to the GPU
        data = data.to(device)
        params = params.to(device)
        labels = labels.to(device)

        # Forward pass of the maping model
        latent_space_reconstruction = model(params)
        # Forward pass of the vae model
        reconstructed_label = vae.decode(latent_space_reconstruction)

        encoded = vae.encode(data)
        mu = vae.MLP_mu(encoded)
        logvar = vae.MLP_logvar(encoded)
        latent_space = mu if vae.beta == 0 else vae.reparametrice(mu, logvar)

        # Calculating the loss function
        # train_loss  = model.loss_function(latent_space_reconstruction, latent_space)
        train_loss  = model.loss_function(reconstructed_label, labels)

        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # save the loss for the epoch
        train_loss_epoch += train_loss.item()

    # Storing the losses in a list for plotting
    train_loss_epoch /= len(train_loader)
    train_losses.append(train_loss_epoch)

    ############################################################################
    #                        Testing of the model                              #
    ############################################################################

    test_loss_epoch = 0
    model.eval()
    with torch.no_grad():
        for (data, labels, params) in tqdm(test_loader, desc = f"Epoch Validation {epoch}/{epochs}", ncols=100):

            data = data.to(device)
            labels = labels.to(device)
            params = params.to(device)

            # Forward pass of the maping model
            latent_space = model(params)
            # Forward pass of the vae model
            reconstructed = vae.decode(latent_space)

            # Calculating the loss function
            test_loss  = model.loss_function(reconstructed, labels)

            # Compute test loss
            test_loss_epoch += test_loss.item()

    # Storing the losses in a list for plotting
    test_loss_epoch /= len(test_loader)
    test_losses.append(test_loss_epoch)

    ############################################################################
    #          Printing and saving the model and loging in wandb               #
    ############################################################################

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
            'hyperparameters': hyperparameters_map,
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
    if (epoch % 25 == 0):
        wandb.watch(model)

############################################################################
#                       Sumarize, save and exit                            #
############################################################################
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
