import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import glob


""" 
Function to load the data from the file, extract the data of the frequencies and the different spectra in intensity
and reshape from (n_x, n_y, n_freq) to (n_x*n_y, n_freq) and normalize it
"""
def load_data(path):
    # open the file containing the parameters data
    with open(f'{path}parameters.pkl', 'rb') as f:
        parameters = pkl.load(f)

    with open(f'{path}profiles.pkl', 'rb') as f:
        profiles = pkl.load(f)
    
    return parameters, profiles


""" 
Function to plot the data of the different spectra in intensity
"""
def plot_data(freq, profiles, color='b', show=False):
    # plot some random spectra in intensity
    for profile in profiles:
        plt.plot(freq, profile, color)

    if show: plt.show()


if __name__ == "__main__":
    # load the data from the file
    folder = sorted(glob.glob('../DATA/neural_he/spectra/*'))[-1] + '/'
    parameters, profiles = load_data(folder)

    # extract a subsample of the data to test
    np.random.seed(777)
    profiles_selec = np.random.randint(0, len(profiles), size=3)
    profiles_selec = [profiles[i]['eta_I'] for i in profiles_selec]

    # extract the frequencies and the profile in eta_I (first profile)
    nus = profiles[0]['nus']
    profiles = [profiles[i]['eta_I'] for i in range(len(profiles))]
    # transform the parameters to save it into the dataset
    params = [[param['B'], param['B_inc'], param['B_inc'],
               param['mu'], param['chi'],
               param['a_voigt'], param['temp'],                                  # Thermal parameters
               param['JKQr'][0][0], param['JKQr'][1][0], param['JKQr'][2][0],
               param['JKQr'][1][1].real, param['JKQr'][2][1].real, param['JKQr'][2][2].real,
               param['JKQr'][1][1].imag, param['JKQr'][2][1].imag, param['JKQr'][2][2].imag,
               param['JKQb'][0][0], param['JKQb'][1][0], param['JKQb'][2][0],
               param['JKQb'][1][1].real, param['JKQb'][2][1].real, param['JKQb'][2][2].real,
               param['JKQb'][1][1].imag, param['JKQb'][2][1].imag, param['JKQb'][2][2].imag] # Radiation field
               for param in parameters]
    params = np.array(params)

    # show the reconstructed spectra using the PCA and the FFT
    plot_data(nus, profiles_selec, color='.b', show=True)

    # create a dictionary with the coefficients of the different models
    # and the instensities that are associated to each model
    models_dict = {
        'profiles'  : profiles,
        'nus'       : nus,
        'parameters': params/np.mean(params, axis=0),
        'norm_param': np.mean(params, axis=0)}

    # save the coefficients to a pkl for training the encoder-decoder network
    with open(f'{folder}model_ready.pkl', 'wb') as f:
        pkl.dump(models_dict, f)
