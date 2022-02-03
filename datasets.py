import pickle as pkl
from random import shuffle
import torch
import numpy as np

# Define the dataset class for storing the data
class spectral_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load the spectral data
        with open(data_path, 'rb') as f:
            data = pkl.load(f)

        # Compute the number of samples for the training and test sets
        self.n_samples = len(data['intensities'])
        self.n_features = len(data['intensities'][0])
        self.n_components = len(data['fft_coeffs'][0])

        # Load the samples (separate the training and test data)
        self.data = np.array(data['intensities'], dtype=np.float32)
        # Load the labels
        self.labels = np.concatenate((data['fft_coeffs'].real, data['fft_coeffs'].imag), 
                                     axis=-1, dtype=np.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples
