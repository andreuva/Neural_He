from turtle import forward
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import MetaLayer
import torch.nn as nn
import torch
from torch_scatter import scatter_mean

from graphnet import EdgeModel



def layer_weight_init(layer, init_type='kaiming'):
    """
    Initialize the weights of a layer.
    :param layer: The layer to initialize.
    :param init_type: The type of initialization to use.
    :return: The initialized layer.
    """
    if isinstance(layer, nn.Linear):
        if init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.xavier_normal_(layer.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_uniform_(layer.weight)
            # nn.init.kaiming_normal_(layer.weight)
        elif init_type == 'normal':
            nn.init.normal_(layer.weight)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
        else:
            raise ValueError('Initialization type not supported.')
        if hasattr(layer, 'bias'):
            layer.bias.data.fill_(0)

    elif isinstance(layer, nn.BatchNorm1d):
        layer.weight.data.fill_(1)
        if hasattr(layer, 'bias'):
            layer.bias.data.fill_(0)


class MLP(nn.Module):
    """
    GENERIC MLP

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int, list
        Number of hidden units in each layer.
    n_hiden_layers : int
        Number of hidden layers.
    output_size : int
        Number of output features.
    activation : str, optional
        Activation function to use. The default is 'relu'.
    layernorm : bool, optional
        Whether to use layer normalization at the end. The default is False.
    """

    def __init__(self, input_size: int, hidden_size, n_hiden_layers: int,  output_size: int, activation='relu', layernorm=False):
        super(MLP, self).__init__()

        # Define the dimensions of the hidden layers
        if(type(hidden_size) is int):
            hidden_size = [hidden_size] * n_hiden_layers
        else:
            assert len(hidden_size) == n_hiden_layers
        
        # Define the activation function
        if (activation == 'relu'):
            self.activation = nn.ReLU()
        elif (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU()
        elif (activation == 'elu'):
            self.activation = nn.ELU()
        elif (activation == 'selu'):
            self.activation = nn.SELU()
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        else:
            raise ValueError('Activation function not supported.')

        # Object to store the model layers in a list
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        self.layers.append(self.activation)

        # Create hidden layers
        for i in range(1, n_hiden_layers):
            self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            self.layers.append(self.activation)
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size[-1], output_size))
        # add the layer normalization if required
        if layernorm:
            self.layers.append(nn.LayerNorm(output_size))
        
        # Define the model
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def get_weights(self):
        # Returns list of weights.
        weights = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.data.numpy())
        return weights

    def weights_init(self, init_type='kaiming'):
        # Initialize the weights of the network.
        for layer in self.layers:
            layer_weight_init(layer, init_type)


class Edge(nn.Module):
    """ 
    Edge Model for updating the edges on the graphNN.

    Parameters
    ----------
    node_latent_size : int
        Number of latent features for each node.
    edge_latent_size : int
        Number of latent features for each edge.
    global_input_size : int
        Number of global input features.
    n_hiden_layers : int
        Number of hidden layers.
    hiden_layers_size : int, list
        Number of hidden units in each layer.
    activation : str, optional
        Activation function to use. The default is 'relu'.
    layernorm : bool, optional
        Whether to use layer normalization at the end. The default is True.
    """
    def __init__(self, node_latent_size: int, edge_latent_size: int, global_input_size: int,
                 n_hiden_layers: int, hiden_layers_size, activation='relu', layernorm=True):
        super(Edge, self).__init__()

        # Define the MLP of the edge model
        self.edge_processor = MLP(input_size=2*node_latent_size+edge_latent_size+global_input_size, 
                            hidden_size=hiden_layers_size,
                            n_hiden_layers=n_hiden_layers,
                            output_size=edge_latent_size,
                            activation=activation,
                            layernorm=layernorm)

    def forward(self, node_i, node_j, edge_ij, global_attr):
        # 
        out = torch.cat([node_i, node_j, edge_ij, global_attr], -1)
        return self.edge_processor(out)
        

class Node(nn.Module):
    """ 
    Node Model for updating the nodes on the graphNN.
    
    Parameters
    ----------
    node_latent_size : int
        Number of latent features for each node.
    edge_latent_size : int
        Number of latent features for each edge.
    global_input_size : int
        Number of global input features.
    n_hiden_layers : int
        Number of hidden layers.
    hiden_layers_size : int, list
        Number of hidden units in each layer.
    activation : str, optional
        Activation function to use. The default is 'relu'.
    layernorm : bool, optional
        Whether to use layer normalization at the end. The default is True.
    """
    def __init__(self, node_latent_size: int, edge_latent_size: int, global_input_size: int,
                 n_hiden_layers: int, hiden_layers_size, activation='relu', layernorm=False):
                
        self.node_processor_1 = MLP(input_size=node_latent_size+edge_latent_size,
                            hidden_size=hiden_layers_size,
                            n_hiden_layers=n_hiden_layers,
                            output_size=node_latent_size,
                            activation=activation,
                            layernorm=layernorm)

        self.node_processor_2 = MLP(input_size=node_latent_size+edge_latent_size+global_input_size,
                            hidden_size=hiden_layers_size,
                            n_hiden_layers=n_hiden_layers,
                            output_size=node_latent_size,
                            activation=activation,
                            layernorm=layernorm)

    def forward(self, nodes, node_edges_index, edge_attr, global_attr):
        # 
        k, j = node_edges_index
        out = torch.cat([nodes[k], edge_attr], -1)
        out = self.node_processor_1(out)
        out = scatter_mean(out, j, dim=0, dim_size=nodes.size(0))
        out = torch.cat([nodes, out, global_attr], -1)
        return self.node_processor_2(out)


class EncodeProcessDecode(nn.Module):
    """ 
    Encode-Process-Decode Network
    paper Learning Mesh-Based Simulation with Graph Networks arXiv:2010.03409

    Parameters
    ----------
    node_input_size : int
        Number of input features for each node.
    edge_input_size : int
        Number of input features for each edge.
    output_size : int
        Number of output features.
    node_latent_size : int
        Number of latent features for each node.
    edge_latent_size : int
        Number of latent features for each edge.
    global_input_size : int
        Number of global input features.
    n_hiden_layers : int
        Number of hidden layers.
    hiden_layers_size : int, list
        Number of hidden units in each layer.
    n_message_passing : int
        Number of message passing steps.
    activation : str, optional
        Activation function to use. The default is 'relu'.
    """
    def __init__(self,
                 node_input_size: int, edge_input_size: int, output_size: int,
                 node_latent_size: int, edge_latent_size: int, global_input_size: int,
                 n_hiden_layers: int, hiden_layers_size, n_message_passing: int,
                 activation='elu'):
        super(EncodeProcessDecode, self).__init__()

        # Copy the parameters to the model attributes
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.output_size = output_size
        self.node_latent_size = node_latent_size
        self.edge_latent_size = edge_latent_size
        self.global_input_size = global_input_size
        self.n_hiden_layers = n_hiden_layers
        self.hiden_layers_size = hiden_layers_size
        self.n_message_passing = n_message_passing
        self.activation = activation

        # Define the MLP of the encoder
        self.edge_encoder = MLP(input_size=node_input_size,
                                hidden_size=hiden_layers_size,
                                n_hiden_layers=n_hiden_layers,
                                output_size=edge_latent_size,
                                activation=activation,
                                layernorm=True)
        
        self.node_encoder = MLP(input_size=node_input_size,
                                hidden_size=hiden_layers_size,
                                n_hiden_layers=n_hiden_layers,
                                output_size=node_latent_size,
                                activation=activation,
                                layernorm=True)
        
        self.encoder_network = Graph(edge_model=self.edge_encoder, node_model=self.node_encoder)

        # Define the MLP of the decoder
        self.decoder_network = MLP(input_size=node_latent_size,
                                   hidden_size=hiden_layers_size,
                                   n_hiden_layers=n_hiden_layers,
                                   output_size=output_size,
                                   activation=activation,
                                   layernorm=True)

        # Define the graph of the processor
        self.processor_network = nn.ModuleList([])
        for i in range(n_message_passing):
            edge_model_i = Edge(node_latent_size=node_latent_size,
                                edge_latent_size=edge_latent_size,
                                global_input_size=global_input_size,
                                n_hiden_layers=n_hiden_layers,
                                hiden_layers_size=hiden_layers_size,
                                activation=activation,
                                layernorm=True)

            node_model_i = Node(node_latent_size=node_latent_size,
                                edge_latent_size=edge_latent_size,
                                global_input_size=global_input_size,
                                n_hiden_layers=n_hiden_layers,
                                hiden_layers_size=hiden_layers_size,
                                activation=activation,
                                layernorm=True)
            
            self.processor_network.append(MetaLayer(edge_model_i,
                                                    node_model_i,
                                                    None))
