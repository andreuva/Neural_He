import torch
import pickle as pkl
from random import shuffle
import numpy as np


# Define the dataset class for storing the data
class profiles_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, train_split=0.85, archiquecture=None):
        # Load the spectral data
        with open(data_path, 'rb') as f:
            data = pkl.load(f)

        # Compute the number of samples for the training and test sets
        self.n_samples = len(data['params'])
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
        self.params = np.array(data['params'][indices], dtype=np.float32)
        self.profiles = np.array(data['profiles'][indices], dtype=np.float32)

        self.labels = self.profiles
        print('Using MLP or CNN architecture (inputs = parameters, labels = profiles)')
        self.data = self.params

        self.n_features = self.data.shape[1]
        self.n_components = self.labels.shape[1]
        self.n_params = self.params.shape[1]
        self.N_nus = len(data['nus'])
        self.nus = data['nus']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __call__(self, index):
        return self.data[index], self.labels[index], self.params[index]

    def __len__(self):
        return self.n_samples


def print_hyperparameters(hyperparameters: dict):
    for key, value in hyperparameters.items():
        if type(value) != dict:
            print(f'{key:<25}: {value}')
        else:
            print(f'{key:<25}: ')
            print_hyperparameters(value)