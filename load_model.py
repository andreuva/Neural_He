import numpy as np
import torch
from dataset import profiles_dataset
from NN import EncoderDecoder
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


# create the dataset to test
dataset = profiles_dataset('../DATA/neural_he/spectra/model_ready_flat_spectrum_100k.pkl', train=False)

# DataLoader is used to load the dataset
test_loader = torch.utils.data.DataLoader(dataset = dataset,
                                          batch_size = 1,
                                          shuffle = True,
                                          pin_memory = True)

# Load the checkpoint and initialize the model
checkpoint = sorted(glob('checkpoints/checkpoint_*.pth'))[-1]
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model = EncoderDecoder(dataset.n_components, dataset.n_features).to(device)
model.load_state_dict(checkpoint['state_dict'])

reconstructed_profiles = []

# predict the test data with the loaded model
model.eval()
with torch.no_grad():
    for (params, fft_coef) in tqdm(test_loader):
        params = params.to(device)
        # fft_coef = fft_coef.to(device)

        prediction = model(params)
        prediction = prediction.detach().cpu().numpy()
        params = params.detach().cpu().numpy()
        # fft_coef = fft_coef.detach().cpu().numpy()

        prediction = np.squeeze(prediction)
        params = np.squeeze(params)
        fft_coef = np.squeeze(fft_coef)

        fft_rec_imag = np.zeros(dataset.n_components, dtype=np.complex64)
        fft_rec_imag.real = prediction[:dataset.n_components]
        fft_rec_imag.imag = prediction[dataset.n_components:]
        fft_rec_imag = fft_rec_imag*dataset.norm_fft
        reconstructed = np.fft.irfft(fft_rec_imag, n=dataset.N_nus)

        fft_imag = np.zeros(dataset.n_components, dtype=np.complex64)
        fft_imag.real = fft_coef[:dataset.n_components]
        fft_imag.imag = fft_coef[dataset.n_components:]
        fft_imag = fft_imag*dataset.norm_fft
        profile = np.fft.irfft(fft_imag, n=dataset.N_nus)

        # save the reconstructed profile to a variable
        reconstructed_profiles.append([params, fft_rec_imag, fft_imag, reconstructed, profile])

profiles = np.array([iterat[-1] for iterat in reconstructed_profiles])
profiles = np.squeeze(profiles)
reconstructions = np.array([iterat[-2] for iterat in reconstructed_profiles])
reconstructions = np.squeeze(reconstructions)
fft_coefs = np.array([iterat[2] for iterat in reconstructed_profiles])
fft_coefs = np.squeeze(fft_coefs)
fft_recons = np.array([iterat[1] for iterat in reconstructed_profiles])
fft_recons = np.squeeze(fft_recons)

for i in range(10):
    plt.plot(profiles[i],'.', label='original')
    plt.plot(reconstructions[i], label='reconstructed')

    plt.legend()
    plt.show()

for i in range(len(profiles)):
    plt.plot(profiles[i], reconstructions[i], ',r')

plt.xscale('log')
plt.yscale('log')
plt.savefig('data_20220324_104926/reconstruction_comparison.png')
plt.close()