import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from dataset import profiles_dataset
from NN import EncoderDecoder
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
print('Checking the GPU availability...')
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
print('-'*50)
print('Creating the training dataset ...')
dataset = profiles_dataset('../DATA/neural_he/spectra/model_ready_flat_spectrum_100k.pkl', train=True)
# create the test dataset
print('Creating the test dataset ...\n')
test_dataset = profiles_dataset('../DATA/neural_he/spectra/model_ready_flat_spectrum_100k.pkl', train=False)

samples_test = set(test_dataset.indices)
samples_train = set(dataset.indices)

# check that the training and test sets are disjoint
print('Number of training samples: {}'.format(len(samples_train)))
print('Number of test samples: {}'.format(len(samples_test)))
print('Number of samples in both sets: {}'.format(len(samples_train.intersection(samples_test))))
assert(len(samples_test.intersection(samples_train)) == 0)
print('Training and test sets are disjoint!\n')

# DataLoader is used to load the dataset
# for training
print('Creating the training DataLoader ...')
train_loader = torch.utils.data.DataLoader(dataset = dataset,
									 batch_size = 256,
									 shuffle = True,
                                     pin_memory = True)

print('Creating the test DataLoader ...')
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 256,
                                          shuffle = True,
                                          pin_memory = True)

# Model Initialization
print('-'*50)
print('Initializing the model ...\n')
model = EncoderDecoder(dataset.n_components, dataset.n_features).to(device)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=333, gamma=0.1)

epochs = 1000
smooth = 0.1
train_losses = []
test_losses = []
best_loss = float('inf')

# start training
print('-'*50)
print('Training the model ...\n')
start_time = time.time()
print('Start time: {}'.format(start_time))
print('-'*50 + '\n')
# for epoch in range(epochs):
for epoch in tqdm(range(epochs), desc=f"Epochs"):
    train_loss_avg = 0
    model.train()
    # for (profiles, fft_coef) in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}", leave = False):
    for (profiles, fft_coef) in train_loader:

        profiles = profiles.to(device)
        fft_coef = fft_coef.to(device)

        # Forward pass
        # Output of Autoencoder
        reconstructed = model(profiles)

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
        # for (profiles, fft_coef) in tqdm(test_loader, desc = "Epoch Validation {epoch}/{epochs}", leave = False):
        for (profiles, fft_coef) in test_loader:

            profiles = profiles.to(device)
            fft_coef = fft_coef.to(device)

            # Forward pass
            # Output of Autoencoder
            reconstructed = model(profiles)

            # Calculating the loss function
            test_loss = loss_function(reconstructed, fft_coef)

            # Compute smoothed loss
            if (test_loss_avg == 0):
                test_loss_avg = test_loss.item()
            else:
                test_loss_avg = smooth * test_loss.item() + (1.0 - smooth) * test_loss_avg

    # Storing the losses in a list for plotting
    test_losses.append(test_loss_avg)

    # print(f"Epoch {epoch}: Train Loss: {train_loss_avg:1.2e}, Test Loss: {test_loss_avg:1.2e}")

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

        # print("Saving best model...")
        filename = f'checkpoint_{time.strftime("%Y%m%d-%H%M%S")}'
        torch.save(checkpoint, 'checkpoints/' + filename + '_best.pth')

    # scheduler.step()

# finished training
end_time = time.time()
print('End time: {}'.format(end_time))
print('Training time: {}'.format(end_time - start_time))
print('-'*50)

print('Saving the model ...')
filename = f'checkpoint_final_{time.strftime("%Y%m%d-%H%M%S")}'
torch.save(checkpoint, 'checkpoints/' + filename + '.pth')
print('Model saved!\n')
print('-'*50)


# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 8), dpi=150)
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(train_losses)
plt.plot(test_losses)
plt.xscale('log')
plt.yscale('log')
plt.savefig(f'checkpoints/losses_{filename}.png')
plt.close()

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig1, ax1 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)

for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25))):
    params, fft_coef = test_dataset[indx]

    fft_imag = np.zeros(test_dataset.n_components, dtype=np.complex64)
    fft_imag.real = fft_coef[:test_dataset.n_components]
    fft_imag.imag = fft_coef[test_dataset.n_components:]
    fft_imag = fft_imag*test_dataset.norm_fft

    profile = np.fft.irfft(fft_imag, n=test_dataset.N_nus)

    fft_rec = model.forward(torch.tensor(params).to(device))
    fft_rec = fft_rec.detach().cpu().numpy()

    fft_rec_imag = np.zeros(test_dataset.n_components, dtype=np.complex64)
    fft_rec_imag.real = fft_rec[:test_dataset.n_components]
    fft_rec_imag.imag = fft_rec[test_dataset.n_components:]
    fft_rec_imag = fft_rec_imag*test_dataset.norm_fft

    reconstructed = np.fft.irfft(fft_rec_imag, n=test_dataset.N_nus)

    ax1.flat[i].plot(fft_coef[:test_dataset.n_components], '--', color='C0')
    ax1.flat[i].plot(fft_coef[test_dataset.n_components:], color='C0')
    ax1.flat[i].plot(fft_rec[:test_dataset.n_components], '--', color='C1')
    ax1.flat[i].plot(fft_rec[test_dataset.n_components:], color='C1')

    ax2.flat[i].plot(test_dataset.nus, profile, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')

fig1.savefig(f'checkpoints/test_fft_{filename}.png', bbox_inches='tight')
plt.close(fig1)

fig2.savefig(f'checkpoints/test_profile_{filename}.png', bbox_inches='tight')
plt.close(fig2)
