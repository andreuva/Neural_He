from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


""" 
Function to load the data from the file, extract the data of the frequencies and the different spectra in intensity
and reshape from (n_x, n_y, n_freq) to (n_x*n_y, n_freq) and normalize it
"""
def load_data(path):
    # open the file containing the parameters data
    with open(f'{path}parameters.pkl', 'rb') as f:
        parameters = pkl.load(f)

    with open(f'{path}profiles.pkl', 'rb') as f:
        profiles = pkl.load(f)
    
    return parameters, profiles


""" 
Function to plot the data of the different spectra in intensity
"""
def plot_data(freq, profiles, color='b', show=False):
    # plot some random spectra in intensity
    for profile in profiles:
        plt.plot(freq, profile, color)

    if show: plt.show()


""" 
Function to compress the spectra extracting the most
important features and reconstructing the spectra
"""
def compress_spectra_pca(freq, intensities, intensities_test, n_components, n_show=3):
    # perform the PCA
    pca = PCA(n_components=n_components)
    pca.fit(intensities)
    # decompose the test spectra
    compressed = pca.transform(intensities_test)
    # reconstruct the spectra
    reconstructed = pca.inverse_transform(compressed)  
    
    return pca, reconstructed


""" 
Function to extract the most importat fourier coefficients of the spectra
and reconstruct the spectra using the fourier coefficients and the frequencies
"""
def compress_spectra_fft(freq, intensities, n_components):
    # perform the fft
    fft = np.fft.rfft(intensities, axis=1)
    # extract the coefficients
    fft_coeff = fft[:,:n_components]
    # reconstruct the spectra
    reconstructed = np.fft.irfft(fft_coeff, n=freq.shape[0], axis=1)
    
    # normalize the data
    # reconstructed = (reconstructed - np.mean(reconstructed))/np.std(reconstructed)

    return fft_coeff, reconstructed

if __name__ == "__main__":
    # load the data from the file
    parameters, profiles = load_data('data_20220322_141552/')

    # extract a subsample of the data to test
    np.random.seed(7777)
    profiles_test = np.random.randint(0, len(profiles), size=3)
    nus = profiles[0]['nus']
    profiles_test = [profiles[i]['eta_I'] for i in profiles_test]
    profiles = [profiles[i]['eta_I'] for i in range(len(profiles))]

    params = [[param['B'], param['ray_out'][0][0], param['ray_out'][0][1],       # Geometry and mag. field
               param['a_voigt'], param['temp'],                                  # Thermal parameters
               param['JKQ_1'][0][0], param['JKQ_1'][1][0], param['JKQ_1'][2][0],
               param['JKQ_1'][1][1].real, param['JKQ_1'][2][1].real, param['JKQ_1'][2][2].real,
               param['JKQ_1'][1][1].imag, param['JKQ_1'][2][1].imag, param['JKQ_1'][2][2].imag] # Radiation field
              for param in parameters]

    # plot the subset
    # plot_data(nus, profiles_test, show=True)

    # compress the data
    pca_object, reconstructed_pca = compress_spectra_pca(nus, profiles, profiles_test, n_components=30)  
    fft_coeff, reconstructed_fft = compress_spectra_fft(nus, profiles_test, 35)

    # show the reconstructed spectra using the PCA and the FFT
    plot_data(nus, profiles_test, color='.b')
    plot_data(nus, reconstructed_fft, color='orange')
    plot_data(nus, reconstructed_pca, color='r', show=True)

    # compute the fft coefficients of all the samples in the data
    fft_coeffs, reconstructed_ffts = compress_spectra_fft(nus, profiles, 50)
    
    # create a dictionary with the coefficients of the different models
    # and the instensities that are associated to each model
    models_dict = {
        'profiles'  : profiles,
        'fft_coeffs': fft_coeffs,
        'nus'       : nus,
        'parameters': params
                  }

    # save the coefficients to a pkl for training the encoder-decoder network
    with open('../DATA/neural_he/spectra/model_profiles.pkl', 'wb') as f:
        pkl.dump(models_dict, f)