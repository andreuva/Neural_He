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

run_loaded = f'checkpoints/trained_model_mlp_eta_I_1M_time_20221125-101654'
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint[len(run_loaded):]}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = '../data/neural_he/spectra/'
readfile = f'model_ready_{coefficient}_{checkpoint["hyperparameters"]["dataset"]}.pkl'
savedir = run_loaded + '/'

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False, archiquecture=archiquecture)
train_dataset = profiles_dataset(f'{readir}{readfile}', train=True, archiquecture=archiquecture)
if archiquecture == 'cnn':
    model = CNN(test_dataset.n_components,  test_dataset.n_features,
                conv_hiden=checkpoint['hyperparameters']['cnn_hidden_size']).to(device)
elif archiquecture == 'mlp':
    model = MLP(test_dataset.n_components,  test_dataset.n_features,
                checkpoint['hyperparameters']['mlp_hidden_size']).to(device)
elif archiquecture == 'bVAE':
    model = bVAE(test_dataset.n_components,  test_dataset.n_features,
                     checkpoint['hyperparameters']['bVAE']).to(device)
else:
    raise ValueError('Architecture not recognized')
model.load_state_dict(checkpoint['state_dict'])

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25))):
    params, profiles = test_dataset[indx]

    reconstructed = model.forward(torch.tensor(params).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(profiles.shape)

    ax2.flat[i].plot(test_dataset.nus, profiles, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}test_profiles.png', bbox_inches='tight')
plt.close(fig2)

print('Ploting and saving Intiensities from the sampled populations from the train data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,train_dataset.n_samples,25))):
    params, profiles = train_dataset[indx]

    reconstructed = model.forward(torch.tensor(params).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(profiles.shape)

    ax2.flat[i].plot(train_dataset.nus, profiles, color='C0')
    ax2.flat[i].plot(train_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}train_profiles.png', bbox_inches='tight')
plt.close(fig2)