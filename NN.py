import torch
from torch import nn
import math
import torch.nn.functional as funct


class MLP(torch.nn.Module):
    def __init__(self, n_components, n_features, hidden_size=[128]):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        modules = []
        modules.append(nn.Linear(n_features, hidden_size[0]))
        modules.append(nn.ReLU())
        for i in range(len(hidden_size) - 1):
            modules.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size[-1], n_components))

        self.Linear = nn.Sequential(*modules)

    def forward(self, x):
        return  self.Linear(x)


class CNN(torch.nn.Module):
    def __init__(self, n_components, n_features,
                       mlp_hiden_in=[64, 128, 256], mlp_hiden_out=[512, 256, 128, 64],
                       conv_hiden=[64], conv_kernel_size=5):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Output dimension for the convolution (Input=64, 3 Maxpooling layers (in/2), 32 output channels)
        self.in_conv = mlp_hiden_in[-1]
        self.out_chanels = conv_hiden[-1]
        self.num_conv = len(conv_hiden)
        self.out_conv = int(self.out_chanels*(self.in_conv/(2**self.num_conv)))

        self.MLP_input = MLP(self.in_conv, n_features, mlp_hiden_in)

        ccn_modules = []
        ccn_modules.append(nn.Conv1d(in_channels=1, out_channels=conv_hiden[0], kernel_size=conv_kernel_size, stride=1, padding=2))
        ccn_modules.append(nn.ReLU())
        ccn_modules.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        for i in range(len(conv_hiden) - 1):
            ccn_modules.append(nn.Conv1d(conv_hiden[i], conv_hiden[i + 1], kernel_size=conv_kernel_size, stride=1, padding=2))
            ccn_modules.append(nn.ReLU())
            ccn_modules.append(torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        self.CNN = torch.nn.Sequential(*ccn_modules)
        self.MLP_output = MLP(n_components, self.out_conv, mlp_hiden_out)
    

    def forward(self, x):
        x = self.MLP_input.forward(x)
        x = x.view(-1, 1, self.in_conv)
        x = self.CNN(x)
        x = x.view(-1, self.out_conv)
        x = self.MLP_output.forward(x)
        return  x

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
        out =  funct.linear(x, self.weight, self.bias)

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
                is_first = is_first))

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, gamma=None, beta=None):
        for layer in self.layers:
            x = layer(x, gamma=gamma, beta=beta)
        return self.last_layer(x)
