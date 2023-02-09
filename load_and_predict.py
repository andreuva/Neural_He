import numpy as np
import torch
from dataset import profiles_dataset
from NN import MLP, CNN
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import pickle as pkl

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

run_loaded = f'checkpoints/trained_model_D3_mlp_eps_Q_time_20230208-144822'
run_loaded = f'checkpoints/trained_model_D3_mlp_eps_U_time_20230208-144752'
run_loaded = f'checkpoints/trained_model_D3_mlp_eps_V_time_20230208-144840'
run_loaded = f'checkpoints/trained_model_D3_mlp_eps_I_time_20230208-144723'

checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = '../data/neural_he/spectra/' # sorted(glob('../DATA/neural_he/spectra/*'))[-2] + '/'
readfile = f'model_ready_D3_{coefficient}_normaliced.pkl'
savedir = run_loaded + '/'

print('Reading data from: ', readir + readfile)
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile}', train=False)

if archiquecture == 'cnn':
    model = CNN(test_dataset.n_features,  test_dataset.n_components,
                conv_hiden=checkpoint['hyperparameters']['cnn_hidden_size']).to(device)
elif archiquecture == 'mlp':
    model = MLP(test_dataset.n_features,  test_dataset.n_components,
                checkpoint['hyperparameters']['mlp_hidden_size']).to(device)
else:
    raise ValueError('Architecture not recognized')
model.load_state_dict(checkpoint['state_dict'])

# predict the emission for the test dataset and plot the results as x (true) vs y (predicted)
print('Predicting the emission for the test dataset and plot the results ...\n')

# compute the predicted emission
analisis = []
for indx in tqdm(range(test_dataset.n_samples), desc = f"sample prediction", ncols=100):
    params, profiles = test_dataset[indx]
    raw, params_raw, norm_coeffs, params_coefs = test_dataset.return_raw_and_normalization(indx)

    reconstructed = model.forward(torch.tensor(params).to(device))
    reconstructed = reconstructed.detach().cpu().numpy()
    reconstructed = reconstructed.reshape(profiles.shape)
    if coefficient == 'eps_I':
        reconstructed = reconstructed*(norm_coeffs[0] - norm_coeffs[1]) + norm_coeffs[2]
        reconstructed = 10**(reconstructed)
    else:
        reconstructed = reconstructed*norm_coeffs[indx]

    error = np.abs(raw - reconstructed)
    analisis.append([raw, reconstructed, error])

analisis = np.array(analisis)
error_perc = analisis[:,2]/analisis[:,0]*100

# plot a histogram of the relative error within the reconstructed sample
# mark the 1, 5 and 10 percentiles with vertical lines
plt.hist(error_perc, bins=500, range=(0, 20))
plt.axvline(np.percentile(error_perc, 90), color='green', label=f'90% --> {np.percentile(error_perc, 90): .2f}')
plt.axvline(np.percentile(error_perc, 95), color='yellow', label=f'95% --> {np.percentile(error_perc, 95): .2f}')
plt.axvline(np.percentile(error_perc, 99), color='red', label=f'99% --> {np.percentile(error_perc, 99): .2f}')
plt.xlabel('Relative error (%)')
plt.ylabel('Number of samples')
plt.title(f'Relative error histogram for the reconstructed {coefficient}')
plt.legend()
plt.xlim(0, 20)
plt.savefig(f'{savedir}relative_error_histogram_{coefficient}.png')
# plt.show()
plt.close()

# plot the predicted output and the ground truth
plt.plot(analisis[:,0], analisis[:,1], '.', alpha=0.1)
plt.plot([analisis[:,:1].min(), analisis[:,:1].max()], [analisis[:,:1].min(), analisis[:,:1].max()], 'k--')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'{coefficient}')
plt.savefig(f'{savedir}predicted_vs_true_{coefficient}.png')
# plt.show()
plt.close()