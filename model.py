import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence


class Pocket2Drug(torch.nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(Pocket2Drug, self).__init__()
        # use a graph neural network as encoder
        self.encoder = JKMCNWMEmbeddingNet(
            num_features=encoder_config['num_features'],
            dim=encoder_config['dim'],
            train_eps=encoder_config['train_eps'],
            num_edge_attr=encoder_config['num_edge_attr'],
            num_layers=encoder_config['num_layers'],
            num_channels=encoder_config['num_channels']
        )

        # use a recurrent neural network as decoder
        self.decoder = RNNDecoder(decoder_config)

    def forward(self, data, smiles, lengths):
        graph_embedding, _, _ = self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        out = self.decoder(graph_embedding, smiles, lengths)

        return out

    def step_rnn(self, int_token, hidden):
        """
        Run one step of the rnn of the decoder.
        Arguments:
        int_token - integer
        hidden - tuple of (h, c)
        """
        x = self.decoder.embedding_layer(
            torch.tensor(int_token).unsqueeze(0).unsqueeze(0))
        x, hidden = self.decoder.rnn(x, hidden)
        x = self.decoder.linear(x)
        x = F.softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        return x.item(), hidden


class RNNDecoder(torch.nn.Module):
    def __init__(self, decoder_config):
        super(RNNDecoder, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=decoder_config['num_embeddings'],
            embedding_dim=decoder_config['embedding_dim'],
            padding_idx=decoder_config['num_embeddings'] - 1
        )

        if decoder_config['which_rnn'] == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=decoder_config['input_size'],
                hidden_size=decoder_config['hidden_size'],
                num_layers=decoder_config['num_layers'],
                batch_first=True,
                dropout=decoder_config['dropout']
            )
        elif decoder_config['which_rnn'] == 'GRU':
            self.rnn = nn.GRU(
                input_size=decoder_config['input_size'],
                hidden_size=decoder_config['hidden_size'],
                num_layers=decoder_config['num_layers'],
                batch_first=True,
                dropout=decoder_config['dropout']
            )

        # softmax output does not include <sos> and <pad>, so
        # decrease the num_embeddings by 2
        self.linear = nn.Linear(
            decoder_config['hidden_size'],
            decoder_config['num_embeddings'] - 2
        )

    def forward(self, graph_embedding, smiles, lengths):
        # Use graph_embedding as input to pre-condition
        # the RNN.
        graph_embedding = graph_embedding.unsqueeze(1)
        # print('graph_embedding: ', graph_embedding.size())

        _, hidden = self.rnn(graph_embedding)
        # print('hidden: ', hidden)

        # feed tokens to embedding layer
        x = self.embedding_layer(smiles)
        # print('x size: ', x.size())

        # pack the padded input
        x = pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # recurrent network, discard (h_n, c_n) in output.
        # Tearcher-forcing is used here, so we directly feed
        # the whole sequence to model.
        x, _ = self.rnn(x, hidden)
        # print('x size: ', x.data.size())

        # linear layer to generate input of softmax
        x = self.linear(x.data)
        # print('x size: ', x.size())
        # print('----------------------------------------')

        # return the packed representation for backpropagation,
        # the targets will also be packed.
        return x


class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on
    Graphs with Jumping Knowledge Networks".
    The GNN layers are now MCNWMConv layer
    """

    def __init__(self,
                 num_features,
                 dim,
                 train_eps,
                 num_edge_attr,
                 num_layers,
                 num_channels=1,
                 layer_aggregate='max'):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        self.conv0 = MCNWMConv(
            in_dim=num_features,
            out_dim=dim,
            num_channels=num_channels,
            num_edge_attr=num_edge_attr,
            train_eps=train_eps
        )
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec('self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)'.format(
                i, num_channels))
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        #x = self.set2set(x, batch)
        return self.set2set(x, batch), x, batch


class MCNWMConv(torch.nn.Module):
    """
    Multi-channel neural weighted message module.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_channels,
                 num_edge_attr=1,
                 train_eps=True,
                 eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(
            Linear(in_dim * num_channels, out_dim),
            LeakyReLU(),
            Linear(out_dim, out_dim)
        )
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(
                nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x


class NWMConv(MessagePassing):
    """
    The neural weighted message (NWM) layer. output of 
    multiple instances of this will produce multi-channel 
    output.
    """

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(Linear(num_edge_attr, 8),
                                  LeakyReLU(),
                                  Linear(8, 1),
                                  ELU())
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        # self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, Tensor):
            x = (x, x)  # x: OptPairTensor

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight
        # generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(edge_nn={})'.format(self.__class__.__name__, self.edge_nn)
