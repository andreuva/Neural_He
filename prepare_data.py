import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import glob
import os


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
    folders = sorted(glob.glob('../data/neural_he/spectra/data_poldeg*'))
    for folder in folders:
        # if the folder is not actually a folder (is a file) move to the next
        if not os.path.isdir(folder):
            continue
        folder = folder + '/'
        print(f'Loading data from {folder}')

        parameters, profiles = load_data(folder)
        params = [[param['B'], param['B_inc'], param['B_az'],
                   param['x'], param['b'], param['h'], param['mu']] for param in parameters]
    
        params = np.array(params)
        nus = profiles[0]['nus']

        components = ['eps_I', 'eps_Q', 'eps_U', 'eps_V']
        for coefficient in components:
            # extract the frequencies and the profile in eta_I (first profile)
            component = np.array([profiles[i][coefficient] for i in range(len(profiles))])

            # create a dictionary with the coefficients of the different models
            # and the instensities that are associated to each model
            models_dict = { 'profiles' : component, 'nus' : nus, 'params' : params,}

            print(f'Saving data to {folder}model_ready_{coefficient}.pkl')
            # save the coefficients to a pkl for training the encoder-decoder network
            with open(f'{folder}model_ready_D3_{coefficient}.pkl', 'wb') as f:
                pkl.dump(models_dict, f)
            
        del models_dict, profiles, parameters, params, nus, component
