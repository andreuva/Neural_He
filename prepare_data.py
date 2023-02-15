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
    sufix = ''
    basedir = '../data/neural_he/spectra/'
    folders = sorted(glob.glob(f'{basedir}data_{sufix}*'))
    for folder in folders:
        # if the folder is not actually a folder (is a file) move to the next
        if not os.path.isdir(folder):
            continue
        folder = folder + '/'
        print(f'Loading data from {folder}')

        try:
            parameters, profiles = load_data(folder)
        except:
            print(f'Error loading data from {folder}')
            continue

        # transform the parameters to save it into the dataset
        params = [[param['B'], param['B_inc'], param['B_az'],
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
        nus = profiles[0]['nus']

        components = ['eps_It', 'eps_Qt', 'eps_Ut', 'eps_Vt', 'eta_It', 'eta_Qt', 'eta_Ut', 'eta_Vt', 'rho_Qt', 'rho_Ut', 'rho_Vt',
                      'eps_Ir', 'eps_Qr', 'eps_Ur', 'eps_Vr', 'eta_Ir', 'eta_Qr', 'eta_Ur', 'eta_Vr', 'rho_Qr', 'rho_Ur', 'rho_Vr',
                      'eps_Ib', 'eps_Qb', 'eps_Ub', 'eps_Vb', 'eta_Ib', 'eta_Qb', 'eta_Ub', 'eta_Vb', 'rho_Qb', 'rho_Ub', 'rho_Vb']
        for coefficient in components:

            component = np.array([profiles[i][coefficient] for i in range(len(profiles))])

            print('Plotting a sample of 100 profiles...')
            np.random.seed(69)
            sample = np.random.randint(0, component.shape[0], 100)
            plt.figure(figsize=(20,20), dpi=200)
            # make a 10x10 grid of plots with random profiles
            for i in range(10):
                for j in range(10):
                    plt.subplot(10,10,i*10+j+1)
                    plt.plot(nus, component[sample[i*10+j]], color = 'blue')
            plt.suptitle(f'Sample of {coefficient}')
            plt.savefig(f'{basedir}prepared_sample_{coefficient}_{folder[-6:-1]}.png')
            # plt.show()
            plt.close()

            # create a dictionary with the coefficients of the different models
            # and the instensities that are associated to each model
            models_dict = { 'profiles' : component, 'nus' : nus, 'params' : params,}

            print(f'Saving data to {folder}model_ready_{coefficient}_{sufix}.pkl')
            # save the coefficients to a pkl for training the encoder-decoder network
            with open(f'{folder}model_ready_{coefficient}_{sufix}.pkl', 'wb') as f:
                pkl.dump(models_dict, f)

        del models_dict, profiles, parameters, params, nus, component
