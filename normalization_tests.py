import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import integrate

with open('../data/neural_he/spectra/data_poldeg_20230224_192007/profiles.pkl', 'rb') as f:
    profiles = pkl.load(f)

with open('../data/neural_he/spectra/data_poldeg_20230224_192007/parameters.pkl', 'rb') as f:
    parameters = pkl.load(f)

nus = np.array([profiles[i]['nus'] for i in range(len(profiles))], dtype=np.float64)
eps_I = np.array([profiles[i]['eps_I'] for i in range(len(profiles))], dtype=np.float64)
eps_Q = np.array([profiles[i]['eps_Q'] for i in range(len(profiles))], dtype=np.float64)
eps_U = np.array([profiles[i]['eps_U'] for i in range(len(profiles))], dtype=np.float64)
eps_V = np.array([profiles[i]['eps_V'] for i in range(len(profiles))], dtype=np.float64)
h = np.array([parameters[i]['h'] for i in range(len(parameters))], dtype=np.float64)/6.957e10
b = np.array([parameters[i]['b'] for i in range(len(parameters))], dtype=np.float64)
x = np.array([parameters[i]['x'] for i in range(len(parameters))], dtype=np.float64)

# select the D3 range
wavelengths = 2.998e8/nus[0]
wavelengths = wavelengths/1e-10
low_mask = wavelengths < 6000
high_mask = wavelengths > 5000
mask = low_mask * high_mask

plt.plot(wavelengths, eps_I[0])
plt.show()

# [b, x, h, mu, chi, B, B_inc, B_az, JKQ]
R_sun = 6.957e10             # solar radius [cm]

# params_normaliced = params.copy()
# params_normaliced = (params_normaliced - params_normaliced.min(axis=0))/(params_normaliced.max(axis=0) - params_normaliced.min(axis=0))

# integrate the profiles in nus and then normalize them as min-max range
eps_I_integrated = eps_I.copy()
eps_Q_integrated = eps_Q.copy()
eps_U_integrated = eps_U.copy()
eps_V_integrated = eps_V.copy()

eps_I_integrated = integrate.simps(eps_I_integrated[:,mask], nus[:,mask], axis=1)
eps_Q_integrated = integrate.simps(eps_Q_integrated[:,mask], nus[:,mask], axis=1)
eps_U_integrated = integrate.simps(eps_U_integrated[:,mask], nus[:,mask], axis=1)
eps_V_integrated = integrate.simps(eps_V_integrated[:,mask], nus[:,mask], axis=1)


plt.plot(h, eps_Q_integrated/eps_I_integrated)
plt.show()

""" 
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
 """
