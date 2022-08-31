import pickle as pkl
import glob
import numpy as np

folder = sorted(glob.glob('../DATA/neural_he/spectra/*'))[-2] + '/'
with open(f'{folder}model_ready.pkl', 'rb') as f:
    data_1 = pkl.load(f)

folder = sorted(glob.glob('../DATA/neural_he/spectra/*'))[-3] + '/'
with open(f'{folder}model_ready.pkl', 'rb') as f:
    data_2 = pkl.load(f)

data_join = {key:np.concatenate((value, value)) for (key,value) in data_1.items()}
[print(len(data_1[key]), len(data_2[key]), len(data_join[key])) for key in data_1.keys()]
[print(data_1[key][0].shape, data_2[key][0].shape, data_join[key][0].shape) for key in data_1.keys()]

with open(f'../DATA/neural_he/spectra/model_ready_500K.pkl', 'wb') as f:
    pkl.dump(data_join, f)
