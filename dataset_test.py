import torch
import pickle as pkl
from random import shuffle
import numpy as np


# Define the dataset class for storing the data
class spectral_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, train_split=0.8):
        # Load the spectral data
        with open(data_path, 'rb') as f:
            data = pkl.load(f)

        # Compute the number of samples for the training and test sets
        self.n_samples = len(data['intensities'])
        if train:
            self.n_samples = int(self.n_samples * train_split)
            start = 0
        else:
            self.n_samples = int(self.n_samples * (1 - train_split))
            start = int(self.n_samples * train_split)

        # create the shuffled indices
        indices = list(range(self.n_samples))
        shuffle(indices)

        # Load the samples (separate the training and test data)
        self.data = np.array(data['intensities'][start:], dtype=np.float32)
        # Load the labels
        self.labels = np.concatenate((data['fft_coeffs'][start:].real, data['fft_coeffs'][start:].imag), 
                                     axis=-1, dtype=np.float32)

        # shuffle the data
        self.data = self.data[indices]
        self.labels = self.labels[indices]

        self.n_features = self.data.shape[1]
        self.n_components = int(self.labels.shape[1]/2)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples