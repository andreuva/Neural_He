import numpy as np
import torch
from dataset_test import spectral_dataset
from NN_test import EncoderDecoder
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
dataset = spectral_dataset('../DATA/neural_he/spectra/model_renormalized_data.pkl', train=False)

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

reconstructed_spectra = []

# predict the test data with the loaded model
model.eval()
with torch.no_grad():
    for (spectra, fft_coef) in tqdm(test_loader):
        spectra = spectra.to(device)
        fft_coef = fft_coef.to(device)

        prediction = model(spectra)
        prediction = prediction.detach().cpu().numpy()
        spectra = spectra.detach().cpu().numpy()
        fft_coef = fft_coef.detach().cpu().numpy()

        prediction = np.squeeze(prediction)
        spectra = np.squeeze(spectra)
        fft_coef = np.squeeze(fft_coef)

        fft_rec_imag = np.zeros(dataset.n_components, dtype=np.complex64)
        fft_rec_imag.real = prediction[:dataset.n_components]
        fft_rec_imag.imag = prediction[dataset.n_components:]
        reconstructed = np.fft.irfft(fft_rec_imag, n=len(spectra))

        # save the reconstructed spectra to a variable
        reconstructed_spectra.append([spectra, reconstructed, fft_coef])

spectres = np.array([iterat[0] for iterat in reconstructed_spectra])
spectres = np.squeeze(spectres)
reconstructions = np.array([iterat[1] for iterat in reconstructed_spectra])
reconstructions = np.squeeze(reconstructions)
fft_coefs = np.array([iterat[2] for iterat in reconstructed_spectra])
fft_coefs = np.squeeze(fft_coefs)

for i in range(10):
    plt.plot(spectres[i],'.', label='original')
    plt.plot(reconstructions[i], label='reconstructed')

    plt.legend()
    plt.show()

for i in range(len(spectres)):
    plt.plot(spectres[i], reconstructions[i], '.r')

plt.xscale('log')
plt.yscale('log')
plt.show()