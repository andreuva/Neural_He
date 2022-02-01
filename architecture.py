import torch
# from torchvision import datasets
# from torchvision import transforms
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl

# Transforms images to a PyTorch Tensor
# tensor_transform = transforms.ToTensor()
# # Download the MNIST Dataset
# dataset = datasets.MNIST(root = "./data",
# 						train = True,
# 						download = True,
# 						transform = tensor_transform)

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

# create the training dataset
dataset = spectral_dataset('data/spectra/model_ready_data.pkl', train=True)
# create the test dataset
test_dataset = spectral_dataset('data/spectra/model_ready_data.pkl', train=False)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
									 batch_size = 128,
									 shuffle = True)

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features
          
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # n_features ==> 18
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU()
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 18 ==> n_components
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_components*2)
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Model Initialization
model = AE(dataset.n_components, dataset.n_features)

# Validation using custom loss function
# def fft_loss(self, out, target):
#     reconstructed = np.fft.irfft(out, n=self.n_features)
#     loss = torch.mean((reconstructed - target)**2)
#     return loss
# loss_function = fft_loss

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-4)

epochs = 2500
losses = []
for epoch in range(epochs):
    for (spectra, fft_coef) in tqdm(loader, desc = "Epoch {}".format(epoch)):
        
        # Forward pass
        # Output of Autoencoder
        reconstructed = model(spectra)
        
        # Calculating the loss function
        loss = loss_function(reconstructed, fft_coef)
        
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Storing the losses in a list for plotting
        losses.append(loss)
  
# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# # Plotting the last 100 values
plt.plot([los.item() for los in losses])
plt.xscale('log')
plt.yscale('log')
plt.show()

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
for indx in np.random.randint(0,test_dataset.n_samples,5):
    spectra, fft_coef = test_dataset[indx]

    fft_rec = model.forward(torch.tensor(spectra))
    fft_rec = fft_rec.detach().numpy()
    fft_rec_imag = np.zeros(test_dataset.n_components, dtype=np.complex64)
    fft_rec_imag.real = fft_rec[:test_dataset.n_components]
    fft_rec_imag.imag = fft_rec[test_dataset.n_components:]
    reconstructed = np.fft.irfft(fft_rec_imag, n=len(spectra))

    plt.plot(spectra, 'or')
    plt.plot(reconstructed, 'b')
    plt.show()
