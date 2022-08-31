from cProfile import run
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


readir = sorted(glob('../DATA/neural_he/spectra/*'))[-2] + '/'
readfile = 'model_ready.pkl'
# create the dataset to test
dataset = profiles_dataset(f'{readir}{readfile}', train=False)
# dataset = profiles_dataset('../DATA/neural_he/spectra/model_ready_flat_spectrum_100k.pkl', train=False)

# DataLoader is used to load the dataset
test_loader = torch.utils.data.DataLoader(dataset = dataset,
                                          batch_size = 1,
                                          shuffle = True,
                                          pin_memory = True)

# Load the checkpoint and initialize the model
run_loaded = sorted(glob('trained_*/'))[1]
print(f'Loading the model from {run_loaded}')
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-1]
print(f'Loading the checkpoint {checkpoint}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model = MLP(dataset.n_components, dataset.n_features).to(device)
model.load_state_dict(checkpoint['state_dict'])

reconstructed_profiles = []

# predict the test data with the loaded model
model.eval()
with torch.no_grad():
    for (params, profile) in tqdm(test_loader):
        params = params.to(device)

        prediction = model(params)
        prediction = prediction.detach().cpu().numpy()
        params = params.detach().cpu().numpy()
        profile = profile.detach().cpu().numpy()

        prediction = np.squeeze(prediction)
        params = np.squeeze(params)
        profile = np.squeeze(profile)

        # save the reconstructed profile to a variable
        reconstructed_profiles.append([params, profile, prediction])

profiles = np.array([iterat[-2] for iterat in reconstructed_profiles])
profiles = np.squeeze(profiles)
reconstructions = np.array([iterat[-1] for iterat in reconstructed_profiles])
reconstructions = np.squeeze(reconstructions)

for i in range(10):
    plt.plot(profiles[i],'.', label='original')
    plt.plot(reconstructions[i], label='reconstructed')

    plt.legend()
    plt.show()

for i in range(len(profiles)):
    plt.plot(profiles[i], reconstructions[i], ',r')

plt.xscale('log')
plt.yscale('log')
plt.savefig('reconstruction_comparison.png')
plt.close()