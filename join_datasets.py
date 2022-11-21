import pickle as pkl
import glob
import numpy as np
import os

print('Loading data...')
for coefficient in ['eta_I', 'eta_Q', 'eta_U', 'eta_V', 'rho_Q', 'rho_U', 'rho_V']:

    data = []
    folders = sorted(glob.glob('../data/neural_he/spectra/data_10M_*'))
    for folder in folders:
        # if the folder is not actually a folder (is a file) move to the next
        if not os.path.isdir(folder):
            continue
        folder = folder + '/'
        print(f'Loading data from {folder}')
        with open(f'{folder}model_ready_10M_{coefficient}.pkl', 'rb') as f:
            data.append(pkl.load(f))

    # data_join = {key:np.concatenate((data_4[key], data_3[key], data_2[key], data_1[key])) for (key,value) in data_1.items()}
    data_join = {}
    data_join['params'] = np.array([data[i]['params'] for i in range(len(data))])
    data_join['profiles'] = np.array([data[i]['profiles'] for i in range(len(data))])
    data_join['nus'] = data[0]['nus']

    [print(f'Length of datasets for key "{key}":',[len(data[i][key]) for i in range(len(data))],f' joint={data_join[key].shape}') for key in data[0].keys()]
    [print(f'Shape of each sample for key "{key}":',[len(data[i][key].shape) for i in range(len(data))],f' joint={data_join[key].shape}') for key in data[0].keys()]

    params = data_join['params']
    params_minmax = (params - params.min(axis=0))/(params.max(axis=0) - params.min(axis=0))
    Jr, Jb = params[:,7:16], params[:,16:]

    # Normalize by J00 (JKQ/J00 is from -1 to 1)
    for i in range(1,9):
        Jr[:,i] = Jr[:,i]/Jr[:,0]
        Jb[:,i] = Jb[:,i]/Jb[:,0]
    # Normalize J00 by doing the log
    Jr[:,0] = np.log10(Jr[:,0])
    Jb[:,0] = np.log10(Jb[:,0])

    params_normaliced = np.zeros_like(params_minmax)
    params_normaliced[:,0:7] = params_minmax[:,0:7].copy()
    params_normaliced[:,7:16] = Jr.copy()
    params_normaliced[:,16:] = Jb.copy()
    data_join['params'] = params_normaliced

    if coefficient == 'eta_I':
        data_join['profiles'] = data_join['profiles']/1e-9
    else:
        data_join['profiles'] = data_join['profiles']/1e-12

    with open(f'../data/neural_he/spectra/model_ready_10M_{coefficient}.pkl', 'wb') as f:
        pkl.dump(data_join, f)

    del data, data_join, params, params_minmax, params_normaliced, Jr, Jb

print('Done!')