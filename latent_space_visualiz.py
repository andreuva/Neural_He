import numpy as np
import torch
from dataset import profiles_dataset
from NN import MLP, CNN, bVAE
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

run_loaded = f'checkpoints/trained_model_bvae_eta_I_bVAE_05_5M_time_20221201-171032'
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint[len(run_loaded):]}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = '../data/neural_he/spectra/'
readfile = f'{checkpoint["hyperparameters"]["dataset"]}'
hyperparameters = checkpoint['hyperparameters']
savedir = run_loaded + '/'

# print the hyperparameters (tabulated correctly) and loss
print('\nHyperparameters')
print('-' * 80)
for key, value in hyperparameters.items():
    if type(value) != dict:
        print(f'{key:<25}: {value}')
    else:
        print(f'{key:<25}: {value[archiquecture]}')
print(f'{"training loss":<25}: {checkpoint["train_loss"]}')
print(f'{"validation loss":<25}: {checkpoint["valid_loss"]}')
print('-' * 80 + '\n')

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False, archiquecture=archiquecture)
# train_dataset = profiles_dataset(f'{readir}{readfile}', train=True, archiquecture=archiquecture)
if archiquecture == 'mlp':
    print('Using MLP')
    mlp_hidden_size = hyperparameters['params'][archiquecture]['mlp_hidden_size']
    model = MLP(test_dataset.n_features, test_dataset.n_components, mlp_hidden_size).to(device)
elif archiquecture == 'cnn':
    print('Using CNN')
    cnn_hidden_size = hyperparameters['params'][archiquecture]['cnn_hidden_size']
    mlp_hiden_in = hyperparameters['params'][archiquecture]['mlp_hiden_in']
    mlp_hiden_out = hyperparameters['params'][archiquecture]['mlp_hiden_out']
    conv_kernel_size = hyperparameters['params'][archiquecture]['conv_kernel_size']
    model = CNN(test_dataset.n_components, test_dataset.n_features, mlp_hiden_in, 
                mlp_hiden_out, cnn_hidden_size, conv_kernel_size).to(device)
elif archiquecture == 'bvae':
    print('Using bVAE')
    bvae_enc_size = hyperparameters['params'][archiquecture]['bvae_enc_size']
    bvae_dec_size = hyperparameters['params'][archiquecture]['bvae_dec_size']
    bvae_latent_size = hyperparameters['params'][archiquecture]['bvae_latent_size']
    bvae_beta = hyperparameters['params'][archiquecture]['bvae_beta']
    model = bVAE(test_dataset.n_components, test_dataset.n_features, bvae_latent_size,
                 bvae_enc_size, bvae_dec_size, bvae_beta).to(device)
else:
    raise ValueError(f'The architecture "{archiquecture}" is not defined')
model.load_state_dict(checkpoint['state_dict'])

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('\nComputing latent space from the sampled populations from the test data ...')
test_latent_samples = []
test_temp_samples = []
for indx in tqdm(range(test_dataset.n_samples)):
    data, labels, params = test_dataset(indx)

    encoded = model.encode(torch.tensor(data).to(device))
    encoded = model.MLP_mu(encoded)
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape((bvae_latent_size))

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
figure, axis = plt.subplots(nrows=bvae_latent_size, ncols=bvae_latent_size, figsize=(30, 20), sharex='col', sharey='row', dpi=200)
for i in tqdm(range(bvae_latent_size)):
    for j in range(bvae_latent_size):
        # print just the points that lay in the percentile 99.8 in x and y
        # this is to avoid the scale issues
        x = test_latent_samples[:, i]
        y = test_latent_samples[:, j]
        # select the indexes between the 0.2 and 99.8 percentile
        x_indx = np.where((x > np.percentile(x, 0.2)) & (x < np.percentile(x, 99.8)))[0]
        y_indx = np.where((y > np.percentile(y, 0.2)) & (y < np.percentile(y, 99.8)))[0]
        # select the indexes that are in both x and y
        indx = np.intersect1d(x_indx, y_indx)
        # plot the points
        axis[i, j].scatter(x[indx], y[indx], c=test_temp_samples[indx], cmap='plasma', s=0.05, alpha=0.15)

print('\nSaving corner plot of the latent space ...')
figure.savefig(savedir + 'latent_space_color_test.png')
