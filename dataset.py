import torch
import pickle as pkl
from random import shuffle
import numpy as np


# Define the dataset class for storing the data
class profiles_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, train_split=0.85):
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
        self.params_raw = np.array(data['params_raw'][indices])
        self.params_norm = np.array(data['params_norm_coeffs'])
        # Load the labels
        self.profiles = np.array(data['profiles'][indices], dtype=np.float32)
        self.profiles_raw = np.array(data['profiles_raw'][indices])
        if 'eps_I' in data_path:
            self.profiles_norm = np.array(data['prof_norm_coeffs'])
        else:
            self.profiles_norm = np.array(data['prof_norm_coeffs'][indices])

        # print('Number of samples: {}'.format(self.n_samples))
        # print('Number of features: {}'.format(self.data.shape[1]))
        # print('Number of components: {}'.format(self.labels.shape[-1]))
        # exit()
        self.n_features = self.params.shape[1]
        self.n_components = 1
        self.N_nus = len(data['nus'])
        self.nus = data['nus']

    def __getitem__(self, index):
        return self.params[index], self.profiles[index]
    
    def return_raw_and_normalization(self, index):
        return self.profiles_raw[index], self.params_raw[index], self.profiles_norm, self.params_norm

    def __len__(self):
        return self.n_samples
