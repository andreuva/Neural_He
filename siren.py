import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as pl
import torch.nn.functional as F

# https://github.com/lucidrains/pi-GAN-pytorch/blob/main/pi_gan_pytorch/pi_gan_pytorch.py

# helper

def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

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

# mapping network
class MappingNetwork(nn.Module):
    def __init__(self, *, dim_input, dim_hidden, dim_out, depth_hidden = 3):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_input, dim_hidden))
        self.layers.append(nn.LeakyReLU(0.2))

        for i in range(depth_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.to_gamma = nn.Linear(dim_hidden, dim_out)
        self.to_beta = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.to_gamma(x), self.to_beta(x)

# siren network
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


if (__name__ == '__main__'):

    dim_in = 2
    dim_hidden = 128
    dim_out = 1
    num_layers = 5
    
    tmp = SirenNet(dim_in = dim_in, dim_hidden = dim_hidden, dim_out = dim_out, num_layers = num_layers)

    print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')
    print((dim_hidden * dim_hidden) * (num_layers - 1) + dim_hidden * (num_layers-1) + dim_hidden * dim_in + dim_hidden + dim_hidden * dim_out + dim_out)

    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)

    xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))
    
    out = tmp(xin).squeeze().reshape((128, 128)).detach().numpy()

    pl.imshow(out)
    pl.show()