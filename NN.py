import torch


class EncoderDecoder(torch.nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Building an linear decoder with Linear
        # layer with Relu activation function and dropout
        self.linear_decoder = torch.nn.Sequential(
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

        # Building a deep convolutional decoder that
        # takes the n_features parameters (1D) and reconstruct the
        # spectra with n_components wavelengths (1D)
        self.conv_decoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 36),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(36, 100),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(8),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            
            torch.nn.Conv1d(8, 16,kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            
            torch.nn.Conv1d(16,64,kernel_size=3, stride=1),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, n_components))

    def forward(self, x):
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return  self.conv_decoder(x)    # decoded


'''
class Net1D(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()

        self.conv1 = nn.Conv1d(1, 8,kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(8, 16,kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16,64,kernel_size=3, stride=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64,2)


    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
 
        return x
'''