import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from dataset_test import spectral_dataset
from NN_test import EncoderDecoder
import time

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


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

# If the nvidia_smi package is installed, then report some additional information
if (NVIDIA_SMI):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
    print(f"Computing in {device} : {nvidia_smi.nvmlDeviceGetName(handle)}")

# create the training dataset
dataset = spectral_dataset('../DATA/neural_he/spectra/model_renormalized_data.pkl', train=True)
# create the test dataset
test_dataset = spectral_dataset('../DATA/neural_he/spectra/model_renormalized_data.pkl', train=False)

# DataLoader is used to load the dataset
# for training
train_loader = torch.utils.data.DataLoader(dataset = dataset,
									 batch_size = 128,
									 shuffle = True,
                                     pin_memory = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 128,
                                          shuffle = True,
                                          pin_memory = True)

# Model Initialization
model = EncoderDecoder(dataset.n_components, dataset.n_features).to(device)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=333, gamma=0.1)

epochs = 2000
smooth = 0.1
train_losses = []
test_losses = []
best_loss = float('inf')

for epoch in range(epochs):
    train_loss_avg = 0
    model.train()
    for (spectra, fft_coef) in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}", leave = False):

        spectra = spectra.to(device)
        fft_coef = fft_coef.to(device)

        # Forward pass
        # Output of Autoencoder
        reconstructed = model(spectra)
        
        # Calculating the loss function
        train_loss = loss_function(reconstructed, fft_coef)
        
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Compute smoothed loss
        if (train_loss_avg == 0):
            train_loss_avg = train_loss.item()
        else:
            train_loss_avg = smooth * train_loss.item() + (1.0 - smooth) * train_loss_avg
        
    # Storing the losses in a list for plotting
    train_losses.append(train_loss_avg)
    
    test_loss_avg = 0
    model.eval()
    with torch.no_grad():
        for (spectra, fft_coef) in tqdm(test_loader, desc = "Epoch Validation {epoch}/{epochs}", leave = False):

            spectra = spectra.to(device)
            fft_coef = fft_coef.to(device)

            # Forward pass
            # Output of Autoencoder
            reconstructed = model(spectra)
            
            # Calculating the loss function
            test_loss = loss_function(reconstructed, fft_coef)

            # Compute smoothed loss
            if (test_loss_avg == 0):
                test_loss_avg = test_loss.item()
            else:
                test_loss_avg = smooth * test_loss.item() + (1.0 - smooth) * test_loss_avg

    # Storing the losses in a list for plotting
    test_losses.append(test_loss_avg)

    print(f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, Test Loss: {test_loss_avg:.6f}")

    if (test_loss_avg < best_loss):
        best_loss = test_loss_avg

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_loss': train_loss_avg,
            'valid_loss': test_loss_avg,
            'best_loss': best_loss,
            # 'hyperparameters': hyperparameters,
            'optimizer': optimizer.state_dict()}

        print("Saving best model...")
        filename = f'checkpoint_{time.strftime("%Y%m%d-%H%M%S")}'
        torch.save(checkpoint, 'checkpoints/' + filename + '_best.pth')

    scheduler.step()

# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# # Plotting the last 100 values
plt.plot(train_losses)
plt.plot(test_losses)
plt.xscale('log')
plt.yscale('log')
plt.show()

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
for indx in np.random.randint(0,test_dataset.n_samples,5):
    spectra, fft_coef = test_dataset[indx]

    fft_rec = model.forward(torch.tensor(spectra).to(device))
    fft_rec = fft_rec.detach().cpu().numpy()
    fft_rec_imag = np.zeros(test_dataset.n_components, dtype=np.complex64)
    fft_rec_imag.real = fft_rec[:test_dataset.n_components]
    fft_rec_imag.imag = fft_rec[test_dataset.n_components:]
    reconstructed = np.fft.irfft(fft_rec_imag, n=len(spectra))

    plt.plot(spectra, 'or')
    plt.plot(reconstructed, 'b')
    plt.show()
