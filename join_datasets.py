import pickle as pkl
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

sufix_database = 'twocomp'
sufix_dataset = 'twocomp'
print('Loading data...')
components = [['eps_It', 'eps_Qt', 'eps_Ut', 'eps_Vt', 'eta_It', 'eta_Qt', 'eta_Ut', 'eta_Vt', 'rho_Qt', 'rho_Ut', 'rho_Vt'],
              ['eps_Ir', 'eps_Qr', 'eps_Ur', 'eps_Vr', 'eta_Ir', 'eta_Qr', 'eta_Ur', 'eta_Vr', 'rho_Qr', 'rho_Ur', 'rho_Vr'],
              ['eps_Ib', 'eps_Qb', 'eps_Ub', 'eps_Vb', 'eta_Ib', 'eta_Qb', 'eta_Ub', 'eta_Vb', 'rho_Qb', 'rho_Ub', 'rho_Vb']]

for component in components:
    for coefficient in component:
        data = []
        base_folder = '../data/neural_he/spectra'
        folders = sorted(glob.glob(f'{base_folder}/data_{sufix_database}*'))[:1]

        for folder in folders:
            # if the folder is not actually a folder (is a file) move to the next
            if not os.path.isdir(folder):
                continue
            folder = folder + '/'
            print(f'Loading data from {folder}')
            try:
                with open(f'{folder}model_ready_{coefficient}_{sufix_database}.pkl', 'rb') as f:
                    data.append(pkl.load(f))
            except:
                print(f'Error loading data from {folder}')
                continue

        print('-'*80+'\n')
        print(f'Joining data from {len(data)} folders...')
        print('-'*80 +'\n')

        data_join = {}
        data_join['params'] = np.concatenate([data[i]['params'] for i in range(len(data))])
        data_join['profiles'] = np.concatenate([data[i]['profiles'] for i in range(len(data))])
        data_join['nus'] = data[0]['nus']

        [print(f'Lengths of "{key}":\n', [data[i][key].shape for i in range(len(data))], f'\n joint={data_join[key].shape}\n') for key in data[0].keys()]
        print('-'*80 +'\n')

        params = data_join['params']
        print('Normalizing parameters...')
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

        print('Nomalizing profiles...')
        if '_I' in coefficient:
            normalization = data_join['profiles'].copy()
            normalization_coefficient = coefficient
            print(f'Update normalization with {coefficient}')
            if 'eps_I' in coefficient:
                data_join['profiles'] = data_join['profiles']/1e-12
            else:
                data_join['profiles'] = data_join['profiles']/1e-9
        else:
            data_join['profiles'] = data_join['profiles']
            data_join[normalization_coefficient] = normalization
            print(f'Normalicing {coefficient} with {normalization_coefficient}: {coefficient}/{normalization_coefficient}')

        # plot the mean and std of the profiles to check how they are distributed
        print('Plotting mean and std of profiles...')
        std = data_join['profiles'].std(axis=0)
        mean = data_join['profiles'].mean(axis=0)
        plt.plot(data_join['nus'], mean, color = 'blue')
        plt.fill_between(data_join['nus'], mean-std, mean+std, color = 'blue', alpha=0.5)
        plt.title(f'Mean of {coefficient}')
        plt.savefig(f'{base_folder}/mean_{coefficient}_{sufix_dataset}.png')
        plt.close()

        # print(f'Saving {coefficient}...')
        # with open(f'{base_folder}/model_ready_{coefficient}_{sufix_dataset}.pkl', 'wb') as f:
        #     pkl.dump(data_join, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f'Cleaning {coefficient}...')
        del data, data_join, params, params_minmax, params_normaliced, Jr, Jb, std, mean
        print('-'*80 +'\n')
        print('-'*80 +'\n')

print('Done!')
