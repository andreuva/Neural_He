import torch


class EncoderDecoder(torch.nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # n_features ==> 18
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU())

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 18 ==> n_components
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_components*2))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
