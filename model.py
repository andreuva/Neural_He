import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from datasets import spectral_dataset
from NN import EncoderDecoder

# create the training dataset
dataset = spectral_dataset('../DATA/neural_he/spectra/model_ready_data.pkl', train=True)
# create the test dataset
test_dataset = spectral_dataset('../DATA/neural_he/spectra/model_ready_data.pkl', train=False)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
									 batch_size = 128,
									 shuffle = True)

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28

# Model Initialization
model = EncoderDecoder(dataset.n_components, dataset.n_features)

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
