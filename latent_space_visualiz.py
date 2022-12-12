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

run_loaded = f'checkpoints/trained_model_bvae_eta_Q_new_aproach_test_time_20221205-123742'
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
print('\nComputing latent space from the sampled populations from the test data ...')
test_latent_samples = []
test_temp_samples = []
for indx in tqdm(range(test_dataset.n_samples), desc='Computing latent space', ncols=50):
    data, labels, params = test_dataset[indx]

    encoded = model.encode(torch.tensor(data).to(device))
    encoded = model.MLP_mu(encoded)
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape((latent_size))

    test_temp_samples.append(params[6])
    test_latent_samples.append(encoded)

test_latent_samples = np.array(test_latent_samples)
test_temp_samples = np.array(test_temp_samples)
test_temp_samples = np.log10(test_temp_samples)
print('shape of the latent space: ', test_latent_samples.shape)

# plot the last two dimensions of the latent space
# dim1 = 27
# dim2 = 4
# plt.figure(figsize=(10, 10), dpi=300)
# plt.scatter(test_latent_samples[:, dim1], test_latent_samples[:, dim2], c=test_temp_samples, cmap='plasma', s=0.05, alpha=0.15)
# plt.colorbar()
# plt.xlabel(f'latent space dimension {dim1}')
# plt.ylabel(f'latent space dimension {dim2}')
# plt.savefig(savedir + f'latent_space_dim{dim1}_dim{dim2}.png')
# plt.close()
# exit()

# create a corner plot of the latent space with the temperature as colorcode
# the temperature is in log scale
print('\nPloting corner plot of the latent space ...')
figure, axis = plt.subplots(nrows=latent_size, ncols=latent_size, figsize=(30, 20), dpi=200)
for i in tqdm(range(latent_size), desc='Plotting', ncols=50):
    for j in range(latent_size):
        axis[i, j].scatter(test_latent_samples[:, j], test_latent_samples[:, i], c=test_temp_samples, cmap='plasma', s=0.5, alpha=0.5)
        axis[i, j].set_xticks([])
        axis[i, j].set_yticks([])

print('\nSaving corner plot of the latent space ...')
figure.savefig(savedir + 'latent_space_color_test.png')
