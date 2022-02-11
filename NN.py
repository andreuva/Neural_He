import torch


class EncoderDecoder(torch.nn.Module):
    def __init__(self, layer_size_encoder, layer_size_decoder):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = torch.nn.ModuleList([])
        self.decoder = torch.nn.ModuleList([])

        for i in range(0,len(layer_size_encoder)-1):
            print(f'Encoder layer {i} : {layer_size_encoder[i]} -> {layer_size_encoder[i+1]}')
            self.encoder.append(torch.nn.Linear(layer_size_encoder[i], layer_size_encoder[i+1]))
            self.encoder.append(torch.nn.ReLU())

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        for i in range(len(layer_size_decoder)-1):
            print(f'Decoder layer {i} : {layer_size_decoder[i]} -> {layer_size_decoder[i+1]}')
            self.decoder.append(torch.nn.Linear(layer_size_decoder[i], layer_size_decoder[i+1]))
            self.decoder.append(torch.nn.ReLU())

    def forward(self, x):
        # Forward pass through the encoder
        for layer in self.encoder:
            x = layer(x)

        # Forward pass through the decoder
        for layer in self.decoder:
            x = layer(x)

        return x
