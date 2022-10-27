import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

with open('../DATA/neural_he/spectra/data_20220901_152322/model_ready_eta_I.pkl', 'rb') as f:
    data_1 = pkl.load(f)

params = data_1['params']

# [B, B_inc, B_az, mu, chi, a_voigt, temp, 
#  Jr00, Jr10, Jr20, Jr11, Jr21, Jr22, Jr11, Jr21, Jr22,
#  Jb00, Jb10, Jb20, Jb11, Jb21, Jb22, Jb11, Jb21, Jb22]

params_normmax = (params/np.abs(params).max(axis=0))
# params_logmax = np.log10(params/np.abs(params).max(axis=0) + 1e-20)
# minmax = (params - params.min(axis=0) + 1e-20)/(np.abs(params).max(axis=0))
# params_logminmax = np.log10(minmax)
params_logmax_sign = np.where(params_normmax <=0, -np.log10(-params_normmax),+np.log10(+params_normmax))
params_logmax_sign_mean = np.where(params_normmax <=0, -np.log10(-params_normmax)+np.log10(np.abs(params_normmax).mean(axis=0)),
                                               +np.log10(+params_normmax)-np.log10(np.abs(params_normmax).mean(axis=0)))
params_minmax = (params - params.min(axis=0))/(params.max(axis=0) - params.min(axis=0))
params_normaliced = params_logmax_sign_mean
params_normaliced[:,0:7] = params_minmax[:,0:7]


# fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
# for i in range(params.shape[1]):
#     ax2.flat[i].hist(params_normmax[:,i], density=True, bins=100)
# plt.savefig('params_normmax.png')
# plt.close()

# fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
# for i in range(params.shape[1]):
#     ax2.flat[i].hist(params_logmax_sign[:,i], density=True, bins=100)
# plt.savefig('params_logmax_sign.png')
# plt.close()

# fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
# for i in range(params.shape[1]):
#     ax2.flat[i].hist(params_logmax_sign_mean[:,i], density=True, bins=100)
# plt.savefig('params_logmax_sign_mean.png')
# plt.close()

# fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
# for i in range(params.shape[1]):
#     ax2.flat[i].hist(params_minmax[:,i], density=True, bins=100)
# plt.savefig('params_minmax.png')
# plt.close()

fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col', dpi=200)
for i in range(params.shape[1]):
    ax2.flat[i].hist(params_normaliced[:,i], density=True, bins=100)
# plt.savefig('params_normaliced.png')
# plt.close()
plt.show()


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
