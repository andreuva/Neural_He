import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import integrate

with open('../data/neural_he/spectra/data_20220901_152322/model_ready_eta_I.pkl', 'rb') as f:
    data_1 = pkl.load(f)

params = data_1['params'].copy()
profiles = data_1['profiles'].copy()
nus = data_1['nus'].copy()

# [b, x, h, mu, chi, B, B_inc, B_az, JKQ]
R_sun = 6.957e10             # solar radius [cm]

params_normaliced = params.copy()
params_normaliced = (params_normaliced - params_normaliced.min(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))

# integrate the profiles in nus and then normalize them as min-max range
profiles_normaliced = profiles.copy()
profiles_normaliced = np.array([integrate.simps(profile, nus) for profile in profiles_normaliced])
profiles_normaliced = profiles_normaliced/10

fig2, ax2 = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col')
for i in range(profiles_normaliced.shape[1]):
    ax2.flat[i].hist(profiles_normaliced[:,i], density=True, bins=1000)
# plt.savefig('params_normaliced.png')
plt.show()
# plt.close()
