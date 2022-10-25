from distutils.command.sdist import sdist
import torch
from torch import nn
import math
import torch.nn.functional as funct


class MLP(torch.nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Building an linear NN
        # with Relu activation function
        self.Linear = torch.nn.Sequential(
            torch.nn.Linear(n_features, 36),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, n_components),
            torch.nn.Identity()
            )

        # Output dimension for the convolution (Input=64, 3 Maxpooling layers (in/2), 32 output channels)
        self.in_conv = 512
        self.out_chanels = 32
        self.num_conv = 5
        self.out_conv = int(self.out_chanels*(self.in_conv/(2**self.num_conv)))

        self.l1 = torch.nn.Linear(n_features, 64)
        self.l2 = torch.nn.Linear(64, 128)
        self.l3 = torch.nn.Linear(128, 256)
        self.l4 = torch.nn.Linear(256, self.in_conv)
        self.conv_layer1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv_layer2 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv_layer3 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv_layer4 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv_layer5 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=64, out_channels=self.out_chanels, kernel_size=5, stride=1, padding=2),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.l5 = torch.nn.Linear(self.out_conv, 256)
        self.l6 = torch.nn.Linear(256, n_components)
        self.LeakyReLU = torch.nn.LeakyReLU()

    def forward(self, x):
        # return  self.Linear(x)
        x = self.l1(x)
        x = self.LeakyReLU(x)
        x = self.l2(x)
        x = self.LeakyReLU(x)
        x = self.l3(x)
        x = self.LeakyReLU(x)
        x = self.l4(x)
        x = self.LeakyReLU(x)
        x = x.view(-1, 1, self.in_conv)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = x.view(-1, self.out_conv)
        x = self.l5(x)
        x = self.LeakyReLU(x)
        x = self.l6(x)
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
