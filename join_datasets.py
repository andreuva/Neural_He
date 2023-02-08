import pickle as pkl
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import glob
import numpy as np
import os
from scipy import integrate

print('Loading data...')
root_dir = '../data/neural_he/spectra/'
for coefficient in ['eps_I', 'eps_Q', 'eps_U', 'eps_V']:

    data = []
    folders = sorted(glob.glob(f'{root_dir}data_D3*'))
    for folder in folders:
        # if the folder is not actually a folder (is a file) move to the next
        if not os.path.isdir(folder):
            continue
        folder = folder + '/'
        print(f'Loading data from {folder}')
        with open(f'{folder}model_ready_D3_{coefficient}.pkl', 'rb') as f:
            data.append(pkl.load(f))

    # data_join = {key:np.concatenate((data_4[key], data_3[key], data_2[key], data_1[key])) for (key,value) in data_1.items()}
    data_join = {}
    data_join['params'] = np.concatenate([data[i]['params'] for i in range(len(data))])
    data_join['profiles'] = np.concatenate([data[i]['profiles'] for i in range(len(data))])
    data_join['nus'] = data[0]['nus']

    # select the D3 range
    wavelengths = 2.998e8/data_join['nus']
    wavelengths = wavelengths/1e-10
    low_mask = wavelengths < 6000
    high_mask = wavelengths > 5000
    mask = low_mask * high_mask

    # [print(f'Length of datasets for key "{key}":',[data[i][key].shape for i in range(len(data))],f' joint={data_join[key].shape}') for key in data_join.keys()]
    # [print(f'Shape of each sample for key "{key}":',[data[i][key].shape for i in range(len(data))],f' joint={data_join[key].shape}') for key in data_join.keys()]

    params_normaliced = data_join['params'].copy()
    # normalize the parameters
    print('Normalizing parameters...')
    normalization_coefficients = {'max': params_normaliced.max(axis=0),
                                  'min': params_normaliced.min(axis=0),
                                  'mean': params_normaliced.mean(axis=0)}
    params_normaliced = (params_normaliced - params_normaliced.mean(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))
    data_join['params'] = params_normaliced
    data_join['params_norm_coeffs'] = normalization_coefficients
    # histogram of the parameters
    # print('Plotting histograms...')
    # if coefficient == 'eps_I':
    #     for i, param in enumerate(['B', 'B_inc', 'B_az', 'x', 'b', 'h', 'mu']):
    #         plt.hist(params_normaliced[:,i], bins=500)
    #         plt.title(param)
    #         plt.show()

    print('integrating profiles...')
    # integrate the profiles in nus and then normalize them as min-max range
    profiles_normaliced = data_join['profiles'].copy()
    # profiles_normaliced = np.trapz(profiles_normaliced, data_join['nus'], axis=1)
    profiles_normaliced = integrate.simps(profiles_normaliced[:,mask], data_join['nus'][mask], axis=1)

    print('Normalizing profiles...')
    if coefficient == 'eps_I':
        eps_I = profiles_normaliced.copy()
        # profiles_normaliced = profiles_normaliced/1e-8
        # profiles_normaliced = 1e-8/(profiles_normaliced+1e-9)
        # profiles_normaliced = (profiles_normaliced - profiles_normaliced.mean())/profiles_normaliced.std()
        profiles_normaliced = np.log10(profiles_normaliced)
        normalization_profile_coefficients = {'max': profiles_normaliced.max(),
                                              'min': profiles_normaliced.min(),
                                              'mean': profiles_normaliced.mean()}
        data_join['prof_norm_coeffs'] = normalization_profile_coefficients
        profiles_normaliced = (profiles_normaliced - profiles_normaliced.mean())/(profiles_normaliced.max() - profiles_normaliced.min())
    else:
        profiles_normaliced = profiles_normaliced/eps_I
        # profiles_normaliced = 1e-11/(profiles_normaliced+1e-13)
        # profiles_normaliced = (profiles_normaliced - profiles_normaliced.mean())/profiles_normaliced.std()
        # profiles_normaliced = np.log10(profiles_normaliced-profiles_normaliced.min()*1.01)
        # profiles_normaliced = (profiles_normaliced - profiles_normaliced.mean())/1e-4/(profiles_normaliced.max() - profiles_normaliced.min())

    # saving the profiles raw and normalized data
    data_join['profiles_raw'] = data_join['profiles'].copy()
    data_join['profiles'] = profiles_normaliced

    # make a mask to remove the outliers (5 and 95 percentile)
    # print('Removing outliers...')
    # mask = (profiles_normaliced > np.percentile(profiles_normaliced, 5)) * (profiles_normaliced < np.percentile(profiles_normaliced, 95))
    # profiles_normaliced_in = profiles_normaliced[mask]
    # profiles_normaliced_out = profiles_normaliced[~mask]

    # histogram of the integrated D3 profiles
    # print('Plotting histograms...')
    # complete histogram
    # plt.hist(profiles_normaliced_in, bins=2000, alpha=0.5)
    # plt.hist(profiles_normaliced_out, bins=2000, alpha=0.5)
    # plt.title(coefficient)
    # plt.show()

    with open(f'{root_dir}model_ready_D3_{coefficient}_normaliced.pkl', 'wb') as f:
        pkl.dump(data_join, f)

    # save the normalization coefficients
    if coefficient == 'eps_I':
        with open(f'{root_dir}profile_normalization_coefficients_D3_{coefficient}.pkl', 'wb') as f:
            pkl.dump(normalization_profile_coefficients, f)
    
    # check if the parameter normalization is already saved
    if not os.path.exists(f'{root_dir}params_normalization_coefficients_D3.pkl'):
        print('Saving parameters normalization coefficients...')
        # save the params normalization coefficients
        with open(f'{root_dir}params_normalization_coefficients_D3.pkl', 'wb') as f:
            pkl.dump(normalization_coefficients, f)

    del data, data_join, params_normaliced

print('Done!')