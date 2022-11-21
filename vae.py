import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from modules import siren

class BetaVAE(nn.Module):
    def __init__(self, 
                dim_in, 
                dim_out, 
                enc_n_hidden, 
                enc_dim_hidden, 
                latent_size, 
                siren_n_hidden, 
                siren_dim_hidden, 
                siren_w0, 
                film_n_hidden, 
                film_dim_hidden,
                beta = 1.0):

        super(BetaVAE, self).__init__()

        self.beta = beta

        #------------------
        # Encoder
        #------------------
        self.encoder = nn.ModuleList([])

        self.encoder.append(nn.Linear(dim_in * dim_out, enc_dim_hidden))
        self.encoder.append(nn.LeakyReLU(0.2))

        for i in range(enc_n_hidden):
            self.encoder.append(nn.Linear(enc_dim_hidden, enc_dim_hidden))
            self.encoder.append(nn.LeakyReLU(0.2))

        self.fc_mu = nn.Linear(enc_dim_hidden, latent_size)
        self.fc_var = nn.Linear(enc_dim_hidden, latent_size)

        for module in self.encoder.modules():
            kaiming_init(module)
        for module in self.fc_mu.modules():
            kaiming_init(module)
        for module in self.fc_var.modules():
            kaiming_init(module)

        #------------------
        # Decoder
        #------------------
        
        # FiLM mapping 
        self.film = siren.MappingNetwork(dim_in = latent_size, 
            dim_hidden = film_dim_hidden, 
            n_hidden = film_n_hidden,
            dim_out = siren_dim_hidden)
        for module in self.film.modules():
            kaiming_init(module)

        # Siren
        self.siren = siren.SirenNet(dim_in = 1, 
            dim_hidden = siren_dim_hidden, 
            dim_out = dim_out, 
            num_layers = siren_n_hidden, 
            w0_initial = siren_w0)

    def encode(self, x):
        n_batch, nz, nvar = x.shape
        x = x.permute(0, 2, 1).reshape(-1, nz*nvar)
        for layer in self.encoder:
            x = layer(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, tau):        
        gamma, beta = self.film(z)
        out = self.siren(tau, gamma, beta)

        return out

    def forward(self, x, tau):        
        mu, logvar = self.encode(x)

        if (self.beta == 0):
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        return self.decode(z, tau), z, mu, logvar

    def loss_function(self, recons, target, mu, log_var):        
        recons_loss = F.mse_loss(recons, target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.beta * kld_loss

        return loss, recons_loss, self.beta * kld_loss

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
