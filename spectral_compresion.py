from select import select
from pyparsing import col
from scipy.__config__ import show
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
    # open the file containing the data (I,Q,U,V and frequency)
    with open(path, 'rb') as f:
        data = pkl.load(f)

    # extract the data of the frequencies and the different spectra in intensity
    freq = data[0]['lambda']
    intensities = data[0]['data'][0]
    # reshape from (n_x, n_y, n_freq) to (n_x*n_y, n_freq)
    intensities = np.reshape(intensities, (intensities.shape[0]*intensities.shape[1], intensities.shape[2]))
    # normalize the data
    intensities = (intensities - np.mean(intensities))/np.std(intensities)

    return freq, intensities


""" 
Function to plot the data of the different spectra in intensity
"""
def plot_data(freq, intensities, color='b', show=False):
    # plot some random spectra in intensity
    
    for spectra in intensities:
        plt.plot(freq, spectra, color)

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
Function to extract the coefficients of the polynomial of degree n
to reconstruct the spectra using the polynomial coefficients and the frequencies
"""
def compress_spectra_poly(freq, intensities, n_components):
    # construct the model
    model = make_pipeline(PolynomialFeatures(n_components), Ridge(alpha=1e-3))
    # reshape the data to feed the model
    freq_fit = np.reshape(freq, (-1,1))
    freq_fit = freq_fit - freq_fit.min()
    reconstructed = []
    models = []

    for spectra in intensities:
        # perform the polynomial regression
        models.append(model.fit(freq_fit, spectra))
        reconstructed.append(model.predict(freq_fit))

    return models, np.array(reconstructed)


"""
Function to extract the coefficients of the spline of degree n
to reconstruct the spectra using the spline coefficients and the frequencies
"""
def compress_spectra_spline(freq, intensities, n_components):
    # construct the model
    model = make_pipeline(SplineTransformer(n_components), Ridge(alpha=1e-3))
    # reshape the data to feed the model
    freq_fit = np.reshape(freq, (-1,1))
    # freq_fit = freq_fit - freq_fit.min()
    reconstructed = []
    models = []

    for spectra in intensities:
        # perform the spline regression
        models.append(model.fit(freq_fit, spectra))
        reconstructed.append(model.predict(freq_fit))

    return models, np.array(reconstructed)

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
    freq, intensities = load_data('data/spectra/raw_data.pkl')

    # extract a subsample of the data to test
    np.random.seed(7777)
    test_selection = np.random.randint(0, intensities.shape[0], size=3)
    intensities_test = intensities[test_selection, :]

    # plot the subset
    plot_data(freq, intensities_test, show=True)
    
    # compress the data
    pca_object, reconstructed_pca = compress_spectra_pca(freq, intensities, intensities_test, n_components=15)  
    poly_models, reconstructed_poly = compress_spectra_poly(freq, intensities_test, 20)
    spline_models, reconstructed_spline = compress_spectra_spline(freq, intensities_test, 20)
    fft_coeff, reconstructed_fft = compress_spectra_fft(freq, intensities_test, 25)

    plot_data(freq, intensities_test, color='.b')
    # plot_data(freq, reconstructed_poly, color='g')
    # plot_data(freq, reconstructed_spline, color='pink')
    plot_data(freq, reconstructed_fft, color='orange')
    plot_data(freq, reconstructed_pca, color='r', show=True)

    # compute the fft coefficients of all the samples in the data
    fft_coeffs, reconstructed_ffts = compress_spectra_fft(freq, intensities, 25)
    
    # create a dictionary with the coefficients of the different models
    # and the instensities that are associated to each model
    models_dict = {
        'intensities': intensities,
        'fft_coeffs': fft_coeffs,
                  }

    # save the coefficients to a pkl for training the encoder-decoder network
    # with open('data/spectra/model_ready_data.pkl', 'wb') as f:
    #     pkl.dump(models_dict, f)