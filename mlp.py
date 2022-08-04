import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as pl

def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLPConditioning(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, activation=nn.Tanh(), bias=True):
        """Simple fully connected network used for conditioning, returning two heads

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True        
        """                
        super(MLPConditioning, self).__init__()

        self.layers = nn.ModuleList([])

        self.activation = activation

        self.layers.append(nn.Linear(n_input, n_hidden, bias=bias))
        self.layers.append(self.activation)

        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
            self.layers.append(self.activation)

        self.gamma = nn.Linear(n_hidden, n_output)
        self.beta = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        gamma = self.gamma(x)
        beta = self.beta(x)

        return gamma, beta

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, activation=nn.Tanh(), bias=True, final_activation=nn.Identity()):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([])

        self.activation = activation
        self.final_activation = final_activation
        
        self.layers.append(nn.Linear(n_input, n_hidden, bias=bias))
        
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        
        self.layers.append(nn.Linear(n_hidden, n_output))
        
    def forward(self, x, gamma=None, beta=None):

        # Apply all layers
        for layer in self.layers[0:-1]:

            # Apply conditioning if present
            if (gamma is not None):
                x = layer(x) * gamma + beta
            else:
                x = layer(x)

            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        
        return x

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)

class MLPMultiFourier(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, mapping_size=128, sigma=[3.0], activation=nn.Tanh(), bias=True, final_activation=nn.Identity()):        
        """Simple fully connected network with random Fourier embedding with several frequencies
        arxiv:2012.10047

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1
        mapping_size : int, optional
            Size of the Fourier embedding applied at the beginning, by default 128
        sigma : list, optional
            List of standard deviations to be used for generating the Gaussian random matrix, by default [3.0]
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """                
        super(MLPMultiFourier, self).__init__()

        self.n_scales = len(sigma)
        self.activation = activation
        self.final_activation = final_activation
                
        # Fourier matrix
        self.sigma = torch.tensor(np.array(sigma).astype('float32'))
        B = torch.randn((self.n_scales, mapping_size, n_input))
        B *= self.sigma[:, None, None]
        
        self.register_buffer("B", B)

        # Layers
        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(2*mapping_size, n_hidden, bias=bias))
        
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        
        self.layers.append(nn.Linear(self.n_scales * n_hidden, n_output))
        
    def forward(self, x, gamma=None, beta=None):

        # Random Fourier encoding                
        tmp = (2. * np.pi * x) @ torch.transpose(self.B, 1, 2) #.t()

        # Compute cosine and sine of the frequencies
        tmp = torch.cat([torch.sin(tmp), torch.cos(tmp)], dim=-1)

        # Apply the neural network to all sigmas simultaneously
        for layer in self.layers[0:-1]:

            # Apply conditioning if available
            if (gamma is not None):
                tmp = layer(tmp) * gamma + beta
            else:
                tmp = layer(tmp)

            tmp = self.activation(tmp)
        
        # Combine all frequencies
        tmp = torch.transpose(tmp, 0, 1).reshape(x.size(0), -1)

        # Final layers
        tmp = self.layers[-1](tmp)
        tmp = self.final_activation(tmp)

        return tmp

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)

if (__name__ == '__main__'):
    
    dim_in = 2
    dim_hidden = 128
    dim_out = 1
    num_layers = 15
    
    tmp = MLPMultiFourier(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=[0.03, 1], activation=nn.ReLU()) #, 0.1, 1.0])
    tmp.weights_init(type='kaiming')

    print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')

    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)

    xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))
    
    out = tmp(xin).squeeze().reshape((128, 128)).detach().numpy()

    pl.imshow(out)
    pl.show()