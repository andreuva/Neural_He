import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import integrate

with open('../data/neural_he/spectra/data_D3_20221123_163349/model_ready_D3_eps_Q.pkl', 'rb') as f:
    data_1 = pkl.load(f)

params = data_1['params'].copy()
profiles = data_1['profiles'].copy()
nus = data_1['nus'].copy()

# select the D3 range
wavelengths = 2.998e8/nus
wavelengths = wavelengths/1e-10
low_mask = wavelengths < 6000
high_mask = wavelengths > 5000
mask = low_mask * high_mask

# [b, x, h, mu, chi, B, B_inc, B_az, JKQ]
R_sun = 6.957e10             # solar radius [cm]

params_normaliced = params.copy()
params_normaliced = (params_normaliced - params_normaliced.min(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))

# integrate the profiles in nus and then normalize them as min-max range
profiles_normaliced = profiles.copy()
profiles_normaliced = integrate.simps(profiles_normaliced[:,mask], nus[mask], axis=1)
profiles_normaliced = profiles_normaliced/1e-9

# select a random sample of the profiles and plot de D3 range
for i in np.random.randint(0, len(profiles_normaliced), 10):
    plt.plot(wavelengths[mask], profiles[i][mask], label=f'integral {profiles_normaliced[i]:.2e}')
plt.legend()
plt.show()


# plot a histogram of the integrated D3 profiles
plt.hist(profiles_normaliced, bins=250, range=(-200,100))
plt.show()

# plot a histogram of the distribution of the parameters (1 for each parameter) in a figure
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        axs[i,j].hist(params_normaliced[:,i*3+j], bins=250, range=(0,1))
plt.show()

