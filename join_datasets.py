import pickle as pkl
import glob
import numpy as np

import pickle as pkl
import glob
import numpy as np
import os

print('Loading data...')
for coefficient in ['eps_I', 'eps_Q', 'eps_U', 'eps_V']:

    data = []
    folders = sorted(glob.glob('../data/neural_he/spectra/data_D3*'))
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

    [print(f'Length of datasets for key "{key}":',[data[i][key].shape for i in range(len(data))],f' joint={data_join[key].shape}') for key in data_join.keys()]
    [print(f'Shape of each sample for key "{key}":',[data[i][key].shape for i in range(len(data))],f' joint={data_join[key].shape}') for key in data_join.keys()]

    params_normaliced = data_join['params'].copy()
    params_normaliced = (params_normaliced - params_normaliced.min(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))
    data_join['params'] = params_normaliced

    # integrate the profiles in nus and then normalize them as min-max range
    profiles_normaliced = data_join['profiles'].copy()
    profiles_normaliced = np.trapz(profiles_normaliced, data_join['nus'], axis=1)
    profiles_normaliced = (profiles_normaliced - profiles_normaliced.min(axis=0))/(profiles_normaliced.max(axis=0) - profiles_normaliced.min(axis=0))

    data_join['profiles'] = profiles_normaliced

    with open(f'../DATA/neural_he/spectra/model_ready_D3_{coefficient}_normaliced.pkl', 'wb') as f:
        pkl.dump(data_join, f)

    del data, data_join, params_normaliced

print('Done!')