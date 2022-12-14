import torch
from torch import nn
import math
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
