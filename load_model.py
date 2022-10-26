import numpy as np
import torch
from dataset import profiles_dataset
from NN import MLP
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

coefficient = 'eta_Q'
archiquecture = 'cnn'
readir = '../DATA/neural_he/spectra/' # sorted(glob('../DATA/neural_he/spectra/*'))[-2] + '/'
readfile = f'model_ready_1M_{coefficient}_normaliced.pkl'
run_loaded = f'trained_model_cnns_eta_Q_bs_256_lr_0.0005_gs_0.55_time_20221025-164830'
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
savedir = run_loaded + '/'

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)

# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model = MLP(test_dataset.n_components, test_dataset.n_features).to(device)
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
fig2.savefig(f'{savedir}test_profile_{run_loaded}_2.png', bbox_inches='tight')
plt.close(fig2)