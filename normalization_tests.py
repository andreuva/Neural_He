import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

with open('../DATA/neural_he/spectra/data_20220901_152322/model_ready_eta_I.pkl', 'rb') as f:
    data_1 = pkl.load(f)

params = data_1['params']
profiles = data_1['profiles']

# [B, B_inc, B_az, mu, chi, a_voigt, temp, 
#  Jr00, Jr10, Jr20, Jr11, Jr21, Jr22, Jr11, Jr21, Jr22,
#  Jb00, Jb10, Jb20, Jb11, Jb21, Jb22, Jb11, Jb21, Jb22]

Jr = params[:,7:16]
Jb = params[:,16:]
for i in range(1,9):
    Jr[:,i] = Jr[:,i]/Jr[:,0]
    Jb[:,i] = Jb[:,i]/Jb[:,0]
Jr[:,0] = np.log10(Jr[:,0])
Jb[:,0] = np.log10(Jb[:,0])

params_minmax = (params - params.min(axis=0))/(params.max(axis=0) - params.min(axis=0))
params_normaliced = np.zeros_like(params_minmax)
params_normaliced[:,0:7] = params_minmax[:,0:7]
params_normaliced[:,7:16] = Jr.copy()
params_normaliced[:,16:] = Jb.copy()

fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col')
for i in range(params.shape[1]):
    ax2.flat[i].hist(params_normaliced[:,i], density=True, bins=1000)
# plt.savefig('params_normaliced.png')
plt.show()
# plt.close()
