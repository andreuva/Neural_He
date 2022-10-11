import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import mlp as mmlp
from dataset import profiles_dataset
from NN import MLP, SirenNet
import time, os, glob
from torchsummary import summary

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


readir = '../DATA/neural_he/spectra/'   # sorted(glob.glob('../DATA/neural_he/spectra/*'))[-1]
readfile = 'model_ready_1M_normaliced.pkl'
print('Reading data from: ', readir + readfile)

# Network params
dim_hidden = 128
layers = 5

batch_size = 256
epochs = 200
learning_rate = 1e-2
step_size_scheduler = 25
gamma_scheduler = 0.1
smooth = 0.2

# construct the base name to save the model
basename = f'trained_model'
savedir = f'./{basename}s_bs_{batch_size}_lr_{learning_rate}_gs_{gamma_scheduler}_time_{time.strftime("%Y%m%d-%H%M%S")}/'
# check if there is a folder for the checkpoints and create it if not
if not os.path.exists(savedir):
    os.makedirs(savedir)

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

# create the training and test dataset
print('-'*50)
print('Creating the training dataset ...')
dataset = profiles_dataset(f'{readir}{readfile}', train=True)
print('Creating the test dataset ...\n')
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)

samples_test = set(test_dataset.indices)
samples_train = set(dataset.indices)

# check that the training and test sets are disjoint
print('Number of training samples: {}'.format(len(samples_train)))
print('Number of test samples: {}'.format(len(samples_test)))
print('Number of samples in both sets: {}'.format(len(samples_train.intersection(samples_test))))
assert(len(samples_test.intersection(samples_train)) == 0)
print('Training and test sets are disjoint!\n')

# DataLoader is used to load the dataset for training and testing
print('Creating the training DataLoader ...')
train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           pin_memory = True)
print('Creating the test DataLoader ...')
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          pin_memory = True)

# Model Initialization
print('-'*50)
print('Initializing the model ...\n')

model = MLP(dataset.n_components,  dataset.n_features).to(device)
# model = mlp.MLPMultiFourier(n_input=dataset.n_features, n_output=dataset.n_components, n_hidden=dim_hidden, n_hidden_layers=layers,
#                             sigma=[5], activation=torch.nn.Tanh(), final_activation=torch.nn.Sigmoid()).to(device)
# model = SirenNet(dim_in=dataset.n_features, dim_hidden=dim_hidden, dim_out=375, num_layers=layers, final_activation=torch.nn.Sigmoid()).to(device)
summary(model, (1, dataset.n_features), batch_size=batch_size)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size_scheduler,
                                            gamma=gamma_scheduler)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-7, total_iters=epochs)

train_losses = []
test_losses = []
best_loss = float('inf')

# start training
print('-'*50)
print('Training the model ...\n')
start_time = time.time()
print('Start time: {}'.format(start_time))
print('-'*50 + '\n')
for epoch in range(epochs):
# for epoch in tqdm(range(epochs), desc=f"Epochs"):
    train_loss_avg = 0
    model.train()
    for (params, profiles) in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}", leave = False):
    # for (params, profiles) in train_loader:

        params = params.to(device)
        profiles = profiles.to(device)

        # Forward pass
        reconstructed = model(params)

        # Calculating the loss function
        train_loss = loss_function(reconstructed, profiles)

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
        for (params, profiles) in tqdm(test_loader, desc = f"Epoch Validation {epoch}/{epochs}", leave = False):
        # for (params, profiles) in test_loader:

            params = params.to(device)
            profiles = profiles.to(device)

            # Forward pass
            # Output of Autoencoder
            reconstructed = model(params)

            # Calculating the loss function
            test_loss = loss_function(reconstructed, profiles)

            # Compute smoothed loss
            if (test_loss_avg == 0):
                test_loss_avg = test_loss.item()
            else:
                test_loss_avg = smooth * test_loss.item() + (1.0 - smooth) * test_loss_avg

    # Storing the losses in a list for plotting
    test_losses.append(test_loss_avg)

    print(f"Epoch {epoch}: Train Loss: {train_loss_avg:1.2e}, Test Loss: {test_loss_avg:1.2e}, lr: {scheduler.get_last_lr()[0]:1.2e}")

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
        filename = f'{basename}_{time.strftime("%Y%m%d-%H%M%S")}'
        torch.save(checkpoint, f'{savedir}' + filename + '.pth')

    scheduler.step()

# finished training
end_time = time.time()
print('End time: {}'.format(end_time))
print('Training time: {}'.format(end_time - start_time))
print('-'*50)

print('Saving the model ...')
filename = f'{basename}_{time.strftime("%Y%m%d-%H%M%S")}_final'
torch.save(checkpoint, f'{savedir}' + filename + '.pth')
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
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f'{savedir}losses_{filename}.png')
plt.close()

# select a random sample from the test dataset and test the network
# then plot the predicted output and the ground truth
print('Ploting and saving Intiensities from the sampled populations from the test data ...\n')
fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)

for i, indx in tqdm(enumerate(np.random.randint(0,test_dataset.n_samples,25))):
    params, profiles = test_dataset[indx]

    reconstructed = model.forward(torch.tensor(params).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()

    ax2.flat[i].plot(test_dataset.nus, profiles, color='C0')
    ax2.flat[i].plot(test_dataset.nus, reconstructed, color='C1')

# saving the plots
print('Saving the plots ...\n')

fig2.savefig(f'{savedir}test_profile_{filename}.png', bbox_inches='tight')
plt.close(fig2)
