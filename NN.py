import torch
from torch import nn
import math
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Building an linear decoder with Linear
        # layer with Relu activation function and dropout
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 36),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(36, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, n_components),
            torch.nn.LeakyReLU())

    def forward(self, x):
        return  self.decoder(x)    # decoded

###############################################################################
#                        SIREN IMPLEMENTATION                                 #
###############################################################################
# Auxiliary functions 
# Check if a variable is defined
def exists(val):
    return val is not None


# Compute the Sinusoidal activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


# Siren layer (Andres Asensio Ramos)
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation
        
        if exists(gamma):
            out = out * gamma

        if exists(beta):
            out = out + beta

        out = self.activation(out)
        return out


# siren network (Andres Asensio Ramos)
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, gamma=None, beta=None):
        for layer in self.layers:
            x = layer(x, gamma=gamma, beta=beta)
        return self.last_layer(x)
