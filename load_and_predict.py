import numpy as np
import torch
from dataset import profiles_dataset, print_hyperparameters
from NN import MLP, CNN
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
else:
    raise ValueError(f'The architecture "{archiquecture}" is not defined')
model.load_state_dict(checkpoint['state_dict'])

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25))):
    data, labels = test_dataset[indx]

    reconstructed = model.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(test_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}test_profiles.png', bbox_inches='tight')

print('Computing and ploting profiles from the sampled populations from the train data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i, indx in tqdm(enumerate(np.random.randint(0,train_dataset.n_samples,25))):
    data, labels = train_dataset[indx]

    reconstructed = model.forward(torch.tensor(data).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(labels.shape)

    ax2.flat[i].plot(train_dataset.nus, labels, color='C0')
    ax2.flat[i].plot(train_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')
fig2.savefig(f'{savedir}train_profiles.png', bbox_inches='tight')