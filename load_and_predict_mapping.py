import numpy as np
import torch
from dataset import profiles_dataset, print_hyperparameters
from NN import bVAE, mapping
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

run_loaded = f'checkpoints/trained_model_bvae_eta_Q_new_aproach_test_time_20221212-082452'
checkpoint_vae = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
checkpoint_map = sorted(glob(f'{run_loaded}/map_*'))[-1]
checkpoint_map = sorted(glob(f'{checkpoint_map}/map_*.pth'))[-2]

# Load the checkpoint and initialize the model
print(f'Loading the vae and mapping models from {run_loaded}')
print(f'Loading the checkpoint {checkpoint_vae[len(run_loaded):]}')
checkpoint_vae = torch.load(checkpoint_vae, map_location=lambda storage, loc: storage)
print(f'Loading the checkpoint {checkpoint_map[len(run_loaded):]}')
checkpoint_map = torch.load(checkpoint_map, map_location=lambda storage, loc: storage)

coefficient = checkpoint_vae['hyperparameters']['coefficient']
readir = checkpoint_vae['hyperparameters']['dataset_dir']
readfile = checkpoint_vae["hyperparameters"]["dataset_file"]
hyperparameters_vae = checkpoint_vae['hyperparameters']
hyperparameters_map = checkpoint_map['hyperparameters']
savedir = run_loaded + '/'

# print the hyperparameters (tabulated correctly) and loss
print('\nHyperparameters')
print('-'*100)
print_hyperparameters(hyperparameters_vae)
print(f'{"training loss":<25}: {checkpoint_vae["train_loss"]}')
print(f'{"validation loss":<25}: {checkpoint_vae["valid_loss"]}')
print('-'*100 + '\n')

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)
train_dataset = profiles_dataset(f'{readir}{readfile}', train=True)

print('Loading the bvae parameters and constructing the model ...')
enc_size = hyperparameters_vae['params']['bvae']['enc_size']
dec_size = hyperparameters_vae['params']['bvae']['dec_size']
latent_size = hyperparameters_vae['params']['bvae']['latent_size']
beta = hyperparameters_vae['params']['bvae']['beta']

# load the model
model_vae = bVAE(test_dataset.n_components, test_dataset.n_features, latent_size, enc_size, dec_size, beta).to(device)
model_vae.load_state_dict(checkpoint_vae['state_dict'])
model_vae.eval()

print('Loading the mapping parameters and constructing the model ...')
hidden_size = hyperparameters_map['hidden_size']

# load the model
model_map = mapping(test_dataset.n_params, latent_size, hidden_size).to(device)
model_map.load_state_dict(checkpoint_map['state_dict'])
model_map.eval()

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25)), ncols=100):
    data, labels, params = test_dataset[indx]

    reconstructed, mu, logvar = model_vae.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(test_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}test_profiles_bvae.png', bbox_inches='tight')

print('Computing and ploting profiles from the sampled populations from the train data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,train_dataset.n_samples,25)), ncols=100):
    data, labels, params = train_dataset[indx]

    reconstructed, mu, logvar = model_vae.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(train_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(train_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}train_profiles_bvae.png', bbox_inches='tight')

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the mapping test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25)), ncols=100):
    data, labels, params = test_dataset[indx]

    latent_space = model_map.forward(torch.tensor(params).to(device))
    reconstructed = model_vae.decode(latent_space)
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(test_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}test_profiles_mapping.png', bbox_inches='tight')

print('Computing and ploting profiles from the sampled populations from the mapping train data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,train_dataset.n_samples,25)), ncols=100):
    data, labels, params = train_dataset[indx]

    latent_space = model_map.forward(torch.tensor(params).to(device))
    reconstructed = model_vae.decode(latent_space)
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(train_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(train_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}train_profiles_mapping.png', bbox_inches='tight')