import torch


class EncoderDecoder(torch.nn.Module):
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
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return  self.decoder(x)    # decoded