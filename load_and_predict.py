import numpy as np
import torch
from dataset import profiles_dataset, print_hyperparameters
from NN import bVAE
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

# check if the GPU is available
cuda = torch.cuda.is_available()
gpu = 0
device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
if cuda:
    print('GPU is available')
    print('Using GPU {}'.format(gpu))
else:
    print('GPU is not available')
    print('Using CPU')
    print(device)

run_loaded = f'checkpoints/'
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint[len(run_loaded):]}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = checkpoint['hyperparameters']['dataset_dir']
readfile = checkpoint["hyperparameters"]["dataset_file"]
hyperparameters = checkpoint['hyperparameters']
savedir = run_loaded + '/'

# print the hyperparameters (tabulated correctly) and loss
print('\nHyperparameters')
print('-' * 80)
print_hyperparameters(hyperparameters)
print(f'{"training loss":<25}: {checkpoint["train_loss"]}')
print(f'{"validation loss":<25}: {checkpoint["valid_loss"]}')
print('-' * 80 + '\n')

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)
train_dataset = profiles_dataset(f'{readir}{readfile}', train=True)

print(f'Using {archiquecture}')
print('Loading the bvae parameters ...')
enc_size = hyperparameters['params']['bvae']['enc_size']
dec_size = hyperparameters['params']['bvae']['dec_size']
latent_size = hyperparameters['params']['bvae']['latent_size']
beta = hyperparameters['params']['bvae']['beta']

# load the model
model = bVAE(test_dataset.n_components, test_dataset.n_features, latent_size, enc_size, dec_size, beta).to(device)
model.load_state_dict(checkpoint['state_dict'])

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25)), ncols=100):
    data, labels, params = test_dataset[indx]

    reconstructed, mu, logvar = model.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(test_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}test_profiles_vae.png', bbox_inches='tight')

print('Computing and ploting profiles from the sampled populations from the train data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,train_dataset.n_samples,25)), ncols=100):
    data, labels, params = train_dataset[indx]

    reconstructed, mu, logvar = model.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(train_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(train_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}train_profiles_vae.png', bbox_inches='tight')