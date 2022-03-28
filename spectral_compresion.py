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
def compress_spectra_pca(freq, profiles, profiles_test, n_components):
    # perform the PCA
    pca = PCA(n_components=n_components)
    pca.fit(profiles)
    # decompose the test spectra
    compressed = pca.transform(profiles_test)
    # reconstruct the spectra
    reconstructed = pca.inverse_transform(compressed)  
    
    return pca, reconstructed


""" 
Function to extract the most importat fourier coefficients of the spectra
and reconstruct the spectra using the fourier coefficients and the frequencies
"""
def compress_spectra_fft(freq, profiles, n_components):
    # perform the fft
    fft = np.fft.rfft(profiles, axis=1)
    # extract the coefficients
    fft_coeff = fft[:,:n_components]
    # reconstruct the spectra
    reconstructed = np.fft.irfft(fft_coeff, n=freq.shape[0], axis=1)
    
    # normalize the data
    # reconstructed = (reconstructed - np.mean(reconstructed))/np.std(reconstructed)

    return fft_coeff, reconstructed


""" 
Function to the spectra in wavelets and reconstruct the spectra using the coefficients
and the frequencies
"""
def compress_spectra_wav(freq, profiles, n_components):

    wav_coeff = freq*0
    reconstructed = wav_coeff*0

    return wav_coeff, reconstructed


if __name__ == "__main__":
    
    """ 
    test = False
    # load the test data
    if test:
        parameters_test, profiles_test = load_data('tests/test_amplitudes_')
        profiles_test = [profiles_test[i]['eta_I'] for i in range(len(profiles_test))]
        params_test = [[param['B'], param['ray_out'][0][0], param['ray_out'][0][1],       # Geometry and mag. field
                        param['a_voigt'], param['temp'],                                  # Thermal parameters
                        param['JKQ_1'][0][0], param['JKQ_1'][1][0], param['JKQ_1'][2][0],
                        param['JKQ_1'][1][1].real, param['JKQ_1'][2][1].real, param['JKQ_1'][2][2].real,
                        param['JKQ_1'][1][1].imag, param['JKQ_1'][2][1].imag, param['JKQ_1'][2][2].imag] # Radiation field
                        for param in parameters_test]
        params_test = np.array(params_test)
        nus = profiles_test[0]['nus']
        fft_coeffs_tests, reconstructed_ffts_test = compress_spectra_fft(nus, profiles_test, 50)
    """
    # load the data from the file
    parameters, profiles = load_data('data_20220322_191501/')

    # extract a subsample of the data to test
    np.random.seed(7777)
    profiles_selec = np.random.randint(0, len(profiles), size=3)
    profiles_selec = [profiles[i]['eta_I'] for i in profiles_selec]

    # extract the frequencies and the profile in eta_I (first profile)
    nus = profiles[0]['nus']
    profiles = [profiles[i]['eta_I'] for i in range(len(profiles))]
    # transform the parameters to save it into the dataset
    params = [[param['B'], param['ray_out'][0][0], param['ray_out'][0][1],       # Geometry and mag. field
               param['a_voigt'], param['temp'],                                  # Thermal parameters
               param['JKQ_1'][0][0], param['JKQ_1'][1][0], param['JKQ_1'][2][0],
               param['JKQ_1'][1][1].real, param['JKQ_1'][2][1].real, param['JKQ_1'][2][2].real,
               param['JKQ_1'][1][1].imag, param['JKQ_1'][2][1].imag, param['JKQ_1'][2][2].imag] # Radiation field
              for param in parameters]
    params = np.array(params)

    # compress the data
    pca_object, reconstructed_pca = compress_spectra_pca(nus, profiles, profiles_selec, n_components=30)  
    fft_coeff, reconstructed_fft = compress_spectra_fft(nus, profiles_selec, 50)
    # wav_coeff, reconstructed_wav = compress_spectra_wav(nus, profiles_selec, 50)

    # show the reconstructed spectra using the PCA and the FFT
    plot_data(nus, profiles_selec, color='.b')
    plot_data(nus, reconstructed_fft, color='orange')
    # plot_data(nus, reconstructed_wav, color='g')
    plot_data(nus, reconstructed_pca, color='r', show=True)

    # compute the fft coefficients of all the samples in the data
    fft_coeffs, reconstructed_ffts = compress_spectra_fft(nus, profiles, 50)

    # create a dictionary with the coefficients of the different models
    # and the instensities that are associated to each model
    models_dict = {
        'profiles'  : profiles,
        'fft_coeffs': fft_coeffs/fft_coeffs.mean(),
        'norm_fft'  : fft_coeffs.mean(),
        'nus'       : nus,
        'parameters': params/np.mean(params, axis=0),
        'norm_param': np.mean(params, axis=0)}

    # save the coefficients to a pkl for training the encoder-decoder network
    with open('../DATA/neural_he/spectra/model_ready_flat_spectrum.pkl', 'wb') as f:
        pkl.dump(models_dict, f)
    '''
    if test:
        models_dict['profiles'] = profiles_test
        models_dict['fft_coeffs'] = fft_coeffs_tests/fft_coeffs.mean()
        models_dict['parameters'] = params_test/np.mean(params, axis=0)
        # save the coefficients to a pkl for testing
        with open('../DATA/neural_he/spectra/model_ready_test_amplitudes.pkl', 'wb') as f:
            pkl.dump(models_dict, f)
    '''