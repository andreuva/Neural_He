import pickle as pkl
import glob
import numpy as np

for coefficient in ['eta_I', 'eta_Q', 'eta_U', 'eta_V', 'rho_Q', 'rho_U', 'rho_V']:

    folder = sorted(glob.glob('../DATA/neural_he/spectra/data*'))[-1] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_1 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/data*'))[-2] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_2 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/data*'))[-3] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_3 = pkl.load(f)

    folder = sorted(glob.glob('../DATA/neural_he/spectra/data*'))[-4] + '/'
    with open(f'{folder}model_ready_{coefficient}.pkl', 'rb') as f:
        data_4 = pkl.load(f)

    # data_join = {key:np.concatenate((data_4[key], data_3[key], data_2[key], data_1[key])) for (key,value) in data_1.items()}
    data_join = {}
    data_join['params'] = np.concatenate((data_4['params'], data_3['params'], data_2['params'], data_1['params']))
    data_join['profiles'] = np.concatenate((data_4['profiles'], data_3['profiles'], data_2['profiles'], data_1['profiles']))
    data_join['nus'] = data_1['nus']

    # [print(f'Length of datasets for key "{key}": l1={len(data_1[key])}, l2={len(data_2[key])}, l3={len(data_3[key])}, l4={len(data_4[key])},\
    #          join={len(data_join[key])}') for key in data_1.keys()]
    # [print(f'Shape of each sample for key "{key}": l1={data_1[key][0].shape}, l2={data_2[key][0].shape}, l3={data_3[key][0].shape},\
    #          l4={data_4[key][0].shape}, join={data_join[key][0].shape}') for key in data_1.keys()]

    params = data_join['params']
    params_normmax = (params/np.abs(params).max(axis=0))
    params_logmax_sign = np.where(params_normmax <=0, -np.log10(-params_normmax)+np.log10(np.abs(params_normmax).mean(axis=0)),
                                                    +np.log10(+params_normmax)-np.log10(np.abs(params_normmax).mean(axis=0)))
    params_minmax = (params - params.min(axis=0))/(params.max(axis=0) - params.min(axis=0))
    params_normaliced = params_logmax_sign
    params_normaliced[:,0:7] = params_minmax[:,0:7]
    data_join['params'] = params_normaliced

    if coefficient == 'eta_I':
        data_join['profiles'] = data_join['profiles']/1e-9
    else:
        data_join['profiles'] = data_join['profiles']/1e-10

    with open(f'../DATA/neural_he/spectra/model_ready_1M_{coefficient}_normaliced.pkl', 'wb') as f:
        pkl.dump(data_join, f)
    
    del data_1, data_2, data_3, data_4, data_join
    del params, params_normmax, params_logmax_sign, params_minmax, params_normaliced
