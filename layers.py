import math

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """
    Self Attention layer used in GCRAM converted to PyTorch. 
    This code was originally implmented in TensorFlow: https://github.com/dalinzhang/GCRAM/blob/master/TfRnnAttention/attention.py
    """

    def __init__(self, hidden_size, attention_size, return_alphas=False):
        super(SelfAttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.return_alphas = return_alphas

        self.w_omega = nn.Parameter(torch.empty(hidden_size, attention_size).normal_(mean=0.0, std=0.1))
        self.b_omega = nn.Parameter(torch.empty(attention_size).normal_(mean=0.0, std=0.1))
        self.u_omega = nn.Parameter(torch.empty(attention_size).normal_(mean=0.0, std=0.1))

    def forward(self, inputs):
        v = torch.tanh(torch.tensordot(inputs, self.w_omega, dims=1) + self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alphas = F.softmax(vu, dim=1)
        output = torch.sum(inputs * alphas.unsqueeze(-1), 1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.hidden_size) + ' -> ' \
               + str(self.attention_size) + ')'


class GraphConvolutionLayer(nn.Module):
    """
    Original GC Layer copied from: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchGraphConvolutionLayer(nn.Module):
    """
    Modified GraphConvolutionLayer for batchwise graph training.
    BatchGraphConvolutionLayer takes batch of graphs as input instead of GraphConvolutionLayer which takes single single graph as input. 
    Input size: [BATCH_SIZE, NUM_CHANNELS, IN_FEATURES] => Output size: [BATCH_SIZE, NUM_CHANNELS, OUT_FEATURES]
    ...
    Attributes
    ----------
    in_features : int
        Number of input features. In PhysioNet EEG MMI dataset's case in_features is number of channels (64).
    out_features : int
        Number of hidden units of GCN layer. 
    """
    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolutionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: [n_samples, n_channels, seq_len]
        # adj: [n_channels, n_channels]
        # weights: [in_features, out_features]

        support = torch.einsum("ijk,kl->ijl", input, self.weight) # [n_samples, n_channels, seq_len] x [in_features, out_features] = [n_samples, n_channels, out_features]
        output = torch.einsum("ij,kjl->kil", adj, support) # [n_channels, n_channels] x [n_samples, n_channels, out_features] = [n_samples, n_channels, out_features]

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
