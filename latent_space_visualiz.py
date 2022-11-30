import numpy as np
import torch
from dataset import profiles_dataset
from NN import MLP, CNN, bVAE
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import corner

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

run_loaded = f'checkpoints/trained_model_bvae_eta_I_VAE_time_20221129-094345'
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
train_dataset = profiles_dataset(f'{readir}{readfile}', train=True, archiquecture=archiquecture)
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
print('Ploting and saving latent space from the sampled populations from the test data ...\n')
test_latent_samples = []
for indx in tqdm(range(test_dataset.n_samples)):
    data, labels = test_dataset[indx]
    encoded = model.encode(torch.tensor(data).to(device))
    encoded = model.MLP_mu(encoded)
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape((bvae_latent_size))
    test_latent_samples.append(encoded)

test_latent_samples = np.array(test_latent_samples)
print('shape of the latent space: ', test_latent_samples.shape)
figure = corner.corner(test_latent_samples) #[:, 0:5])
figure.savefig(savedir + 'latent_space_test.png')
plt.close(figure)

# Do the same for the training dataset
print('Ploting and saving latent space from the sampled populations from the training data ...\n')
train_latent_samples = []
for indx in tqdm(range(train_dataset.n_samples)):
    data, labels = train_dataset[indx]
    encoded = model.encode(torch.tensor(data).to(device))
    encoded = model.MLP_mu(encoded)
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape((bvae_latent_size))
    train_latent_samples.append(encoded)

train_latent_samples = np.array(train_latent_samples)
print('shape of the latent space: ', train_latent_samples.shape)
figure = corner.corner(train_latent_samples) #[:, 0:5])
figure.savefig(savedir + 'latent_space_train.png')
plt.close(figure)