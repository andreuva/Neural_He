import pickle as pkl
import glob
import numpy as np
import scipy as sp

print('Loading data...')
for coefficient in ['eps_I', 'eps_Q', 'eps_U', 'eps_V']:

    folder = sorted(glob.glob('../DATA/neural_he/spectra/D3_data*'))[-1] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_1 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/D3_data*'))[-2] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_2 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/D3_data*'))[-3] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_3 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/D3_data*'))[-4] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_4 = pkl.load(f)

    # data_join = {key:np.concatenate((data_4[key], data_3[key], data_2[key], data_1[key])) for (key,value) in data_1.items()}
    data_join = {}
    data_join['params'] = np.concatenate((data_4['params'], data_3['params'], data_2['params'], data_1['params']))
    data_join['profiles'] = np.concatenate((data_4['profiles'], data_3['profiles'], data_2['profiles'], data_1['profiles']))
    data_join['nus'] = data_1['nus']

    [print(f'Length of datasets for key "{key}": l1={len(data_1[key])}, l2={len(data_2[key])}, l3={len(data_3[key])}, l4={len(data_4[key])},\
             join={len(data_join[key])}') for key in data_1.keys()]
    [print(f'Shape of each sample for key "{key}": l1={data_1[key][0].shape}, l2={data_2[key][0].shape}, l3={data_3[key][0].shape},\
             l4={data_4[key][0].shape}, join={data_join[key][0].shape}') for key in data_1.keys()]

    params_normaliced = data_join['params'].copy()
    params_normaliced = (params_normaliced - params_normaliced.min(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))
    data_join['params'] = params_normaliced

    # integrate the profiles in nus and then normalize them as min-max range
    profiles_normaliced = data_join['profiles'].copy()
    profiles_normaliced = np.array([sp.integrate.simps(profile, data_join['nus']) for profile in profiles_normaliced])
    profiles_normaliced = (profiles_normaliced - profiles_normaliced.min(axis=0))/(profiles_normaliced.max(axis=0) - profiles_normaliced.min(axis=0))

    data_join['profiles'] = profiles_normaliced

    with open(f'../DATA/neural_he/spectra/modelD3_ready_1M_{coefficient}_normaliced.pkl', 'wb') as f:
        pkl.dump(data_join, f)

    del data_1, data_2, data_3, data_4, data_join
    del params_normaliced, profiles_normaliced

print('Done!')