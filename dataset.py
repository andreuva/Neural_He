import torch
import pickle as pkl
from random import shuffle
import numpy as np


# Define the dataset class for storing the data
class profiles_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, train_split=0.8):
        # Load the spectral data
        with open(data_path, 'rb') as f:
            data = pkl.load(f)

        # Compute the number of samples for the training and test sets
        self.n_samples = len(data['parameters'])
        if train:
            start = 0
            self.n_samples = int(self.n_samples * train_split)
        else:
            start = int(self.n_samples * train_split)
            self.n_samples = int(self.n_samples * (1 - train_split))

        # create the shuffled indices
        indices = list(range(start, start+self.n_samples))
        shuffle(indices)
        self.indices = indices

        # Load the samples (separate the training and test data)
        self.data = np.array(data['parameters'][indices], dtype=np.float32)
        # Load the labels
        profiles = np.array(data['profiles'])
        self.labels = np.array(profiles[indices], dtype=np.float32)

        # add the normalization factors to the dataset object
        self.profile_norm = 1e-10
        self.norm_param = data['norm_param']
        self.labels = self.labels/self.profile_norm

        self.n_features = self.data.shape[1]
        self.n_components = self.labels.shape[1]
        self.N_nus = len(data['nus'])
        self.nus = data['nus']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples