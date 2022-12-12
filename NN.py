import torch
from torch import nn
import torch.nn.functional as funct


class MLP(torch.nn.Module):
    """ 
    Class for a simple fully connected NN with LeakyReLU activations
    """
    def __init__(self, n_features, n_components, hidden_size=[128]):
        """
        PARAMETERS
        ----------
        n_features: int
            Number of parameters (features) in the dataset
        n_components: int
            Number of points in the wavelength grid (label points)
        hidden_size: list
            List of integers with the number of neurons in each hidden layer
        """
        # Initialize the parent class
        super().__init__()
        # save the parameters
        self.n_components = n_components
        self.n_features = n_features
        # create the layers appending them to the list
        modules = []
        modules.append(nn.Linear(n_features, hidden_size[0]))
        modules.append(nn.LeakyReLU())
        for i in range(len(hidden_size) - 1):
            modules.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(hidden_size[-1], n_components))
        # create the model
        self.Linear = nn.Sequential(*modules)

    def forward(self, x):
        return  self.Linear(x)


class CNN(torch.nn.Module):
    """ 
    Class for a mixed 1D MLP and convolutional NN with LeakyReLU activations
    INPUT --> MLP --> CNN --> MLP --> OUTPUT
    """
    def __init__(self, n_components, n_features,
                       mlp_hiden_in=[64, 128, 256], mlp_hiden_out=[512], 
                       conv_hiden=[64], conv_kernel_size=5):
        """
        PARAMETERS
        ----------
        n_components: int
            Number of points in the wavelength grid (label points)
        n_features: int
            Number of parameters (features) in the dataset
        mlp_hiden_in: list
            List of integers with the number of neurons in each hidden layer of the initial MLP
        mlp_hiden_out: list
            List of integers with the number of neurons in each hidden layer of the final MLP
        conv_hiden: list
            List of integers with the number of neurons in each hidden layer of the convolutional NN
        conv_kernel_size: int
            Size of the kernel in the convolutional NN
        """
        # Initialize the parent class
        super().__init__()

        # save the parameters
        self.n_components = n_components
        self.n_features = n_features
        # Output dimension for the convolution (Input=64, 3 Maxpooling layers (in/2), 32 output channels)
        self.in_conv = mlp_hiden_in[-1]
        self.out_chanels = conv_hiden[-1]
        self.num_conv = len(conv_hiden)
        # compute the output dimension of the convolutional NN
        self.out_conv = int(self.out_chanels*(self.in_conv/(2**self.num_conv)))
        
        # create the input MLP calling the class MLP
        self.MLP_input = MLP(n_features, self.in_conv, mlp_hiden_in)

        # create the convolutional NN
        ccn_modules = []
        ccn_modules.append(nn.Conv1d(in_channels=1, out_channels=conv_hiden[0], kernel_size=conv_kernel_size, stride=1, padding=2))
        ccn_modules.append(nn.LeakyReLU())
        ccn_modules.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        for i in range(len(conv_hiden) - 1):
            ccn_modules.append(nn.Conv1d(conv_hiden[i], conv_hiden[i + 1], kernel_size=conv_kernel_size, stride=1, padding=2))
            ccn_modules.append(nn.LeakyReLU())
            ccn_modules.append(torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        # convert the list of modules to a sequential model
        self.CNN = torch.nn.Sequential(*ccn_modules)
        # create the final MLP calling the class MLP
        self.MLP_output = MLP(self.out_conv, n_components, mlp_hiden_out)
    
    # define the forward pass
    def forward(self, x):
        # pass the input through the MLP
        x = self.MLP_input.forward(x)
        # reshape the input for the convolutional NN
        x = x.view(-1, 1, self.in_conv)
        # pass the input through the convolutional NN
        x = self.CNN(x)
        # reshape the output of the convolutional NN for the MLP
        x = x.view(-1, self.out_conv)
        # pass the input through the MLP
        x = self.MLP_output.forward(x)
        return  x


# map the data to the latent space
class mapping(torch.nn.Module):
    """
    Class for a mapping from the parameter space to the latent space
    """
    def __init__(self, n_features, latent_dim=25, hidden_size=[256, 128, 64, 32]):
        """
        PARAMETERS
        ----------
        n_features: int
            Number of parameters (features) in the dataset
        latent_dim: int
            Number of dimensions in the latent space
        hidden_size: list
            List of integers with the number of neurons in each hidden layer of the encoder and decoder (shared)
        """
        # save the parameters of the network
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Initialize the parent class
        super().__init__()

        # create the encoder
        self.encoder = MLP(n_features, latent_dim, hidden_size=hidden_size)

    # define the encoder
    def map(self, x):
        return self.encoder(x)

    # define the forward pass
    def forward(self, x):
        z = self.map(x)
        return z
    
    def loss_function(self, recons, target):
        """
        Loss function for the mapping
        PARAMETERS
        ----------
        recons: torch.tensor
            Reconstructed parameters
        target: torch.tensor
            True parameters
        """
        # compute the MSE loss
        return funct.mse_loss(recons, target)


# constrained variational autoencoder
class bVAE(torch.nn.Module):
    """
    Class for a constrained variational autoencoder with LeakyReLU activations and a Gaussian prior on the latent space 
    """
    def __init__(self, n_components, n_features, latent_dim=25, 
                 encoder_size=[256, 128, 64, 32], decoder_size=[32, 64, 128, 256], beta=0.0):
        """
        PARAMETERS
        ----------
        n_components: int
            Number of points in the wavelength grid (label points)
        n_features: int
            Number of parameters (features) in the dataset
        latent_dim: int
            Number of dimensions in the latent space
        hidden_size: list
            List of integers with the number of neurons in each hidden layer of the encoder and decoder (shared)
        beta: float
            Weight of the KL divergence in the loss function
        """
        # save the parameters of the network
        self.n_components = n_components
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.beta = beta

        # Initialize the parent class
        super().__init__()

        # create the encoder
        self.encoder = MLP(n_components, encoder_size[-1], hidden_size=encoder_size)
        # create the mean and variance layers
        self.MLP_mu = torch.nn.Linear(encoder_size[-1], latent_dim)
        self.MLP_logvar = torch.nn.Linear(encoder_size[-1], latent_dim)
        # create the decoder
        self.decoder = MLP(latent_dim, n_components, hidden_size=decoder_size)
        self.mapping = mapping(n_features, latent_dim=latent_dim, hidden_size=encoder_size)

    # define the encoder
    def encode(self, x):
        return self.encoder(x)

    # define the reparametrization trick
    def reparametrice(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # define the decoder
    def decode(self, z):
        return self.decoder(z)

    # define the forward pass
    def forward(self, x):
        encoded = self.encode(x)
        mu = self.MLP_mu(encoded)
        logvar = self.MLP_logvar(encoded)

        if self.beta == 0:
            z = mu
        else:
            z = self.reparametrice(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

    # define the loss function
    def loss_function(self, recons, target, mu, logvar):
        """
        Compute the loss function for the constrained variational autoencoder

        PARAMETERS
        ----------
        recons: torch.tensor
            Reconstructed data
        target: torch.tensor
            Target data
        mu: torch.tensor
            Mean of the latent space
        logvar: torch.tensor
            Logarithm of the variance of the latent space
        """
        # compute the reconstruction loss
        recon_loss = funct.mse_loss(recons, target)
        # compute the KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        return recon_loss + self.beta * kld_loss, recon_loss, kld_loss*self.beta
