import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np


def padding_mask_k(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x 0]     [[0 0 0 1]
     [x x x 0]->    [0 0 0 1]
     [x x x 0]]     [0 0 0 1]] uint8
    """
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask

def padding_mask_q(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x x]      [[0 0 0 0]
     [x x x x]  ->   [0 0 0 0]
     [0 0 0 0]]      [1 1 1 1]] uint8
    """
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.4, alpha=0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        adj: (N, N)
        h: (N, in_features)
        '''
        h = h.unsqueeze(0).to('cuda') # 1,13,769
        adj = adj.unsqueeze(0).to('cuda')
        Wh = torch.bmm(h, (self.W).unsqueeze(0).repeat(h.size(0),1,1))
        e = self._prepare_attentional_mechanism_input(Wh).to('cuda')

        zero_vec = -9e15*torch.ones_like(e).to('cuda')

        attention = torch.where(adj > 0, e, zero_vec)#[N,N]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)#[N,N].[N,out_features]=>[N,out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # 先分别与a相乘再进行拼接
        n=Wh.size(1)
        Wh1 = torch.bmm(Wh, (self.a[:self.out_features, :]).unsqueeze(0).repeat(Wh.size(0),1,1))
        Wh2 = torch.bmm(Wh, (self.a[:self.out_features, :]).unsqueeze(0).repeat(Wh.size(0),1,1))

        # broadcast add
        e = Wh1.repeat(1,1,n) + Wh2.permute(0,2,1).repeat(1,n,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight.unsqueeze(
            0).expand(input.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand(input.shape[0], -1, -1)
        if self.skip:
            output += support

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class SingleAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )

        self.linear_final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )


        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, q, k, v, scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(q, k)
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        scale = v.size(-1)**-0.5

        attention = torch.bmm(q, k.transpose(-2, -1))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        # attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        output = self.linear_final(output)
        output = self.layer_norm(output + q)

        return output


class SingleAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                SingleAttentionLayer(hidden_size, dropout_p)
                for _ in range(n_layers)
            ])

    def forward(self, q, v):
        attn_mask = padding_mask_k(q, v)
        softmax_mask = padding_mask_q(q, v)

        for encoder in self.encoder_layers:
            q = encoder(q, v, v, attn_mask=attn_mask, softmax_mask=softmax_mask)

        return q

class Graph(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dropout):
        super(Graph, self).__init__()
        self.fc_k = nn.Linear(dim_in, dim_hidden)
        self.fc_q = nn.Linear(dim_in, dim_hidden)

        dim_hidden = dim_out if num_layers == 1 else dim_hidden
        self.layers = nn.ModuleList([
            #GraphConvolution(dim_in, dim_hidden)
            GraphConvolution(dim_in, dim_hidden),

        ])

        for i in range(num_layers - 1):
            dim_tmp = dim_out if i == num_layers - 2 else dim_hidden
            self.layers.append(GraphConvolution(dim_hidden, dim_tmp))

        self.dropout = dropout



    def build_graph(self, x):
        batch_size, s_len = x.shape[0], x.shape[1]
        emb_k = self.fc_k(x)
        emb_q = self.fc_q(x)
        length = torch.tensor([s_len] * batch_size, dtype=torch.long)

        s = torch.bmm(emb_k, emb_q.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(length, length)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))

        A = s  # F.softmax(s, dim=2)  # [B, t1, t2]

        # remove nan from softmax on -inf
        A.data.masked_fill_(A.data != A.data, 0)

        return A

    def forward(self, X, A):
        for layer in self.layers:
            X = F.relu(layer(X, A))
            X = F.dropout(X, self.dropout, training=self.training)
        return X




import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
  def __init__(self, rnn_model):
    super().__init__()
    self.rnn_model = rnn_model

  def forward(self, seq_input, seq_lens, initial_state=None):
    """A wrapper over pytorch's rnn to handle sequences of variable length.

    Arguments
    ---------
    seq_input : torch.Tensor
        Input sequence tensor (padded) for RNN model.
        Shape: (batch_size, max_sequence_length, embed_size)
    seq_lens : torch.LongTensor
        Length of sequences (b, )
    initial_state : torch.Tensor
        Initial (hidden, cell) states of RNN model.

    Returns
    -------
        Single tensor of shape (batch_size, rnn_hidden_size) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence.
    """
    max_sequence_length = seq_input.size(1)
    sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
    sorted_seq_input = seq_input.index_select(0, fwd_order)
    packed_seq_input = pack_padded_sequence(
      sorted_seq_input, lengths=sorted_len, batch_first=True
    )

    if initial_state is not None:
      hx = initial_state
      assert hx[0].size(0) == self.rnn_model.num_layers
    else:
      sorted_hx = None

    self.rnn_model.flatten_parameters()

    outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, sorted_hx)

    # pick hidden and cell states of last layer
    h_n = h_n[-1].index_select(dim=0, index=bwd_order)
    c_n = c_n[-1].index_select(dim=0, index=bwd_order)

    outputs = pad_packed_sequence(
      outputs, batch_first=True, total_length=max_sequence_length
    )[0].index_select(dim=0, index=bwd_order)

    return outputs, (h_n, c_n)

  @staticmethod
  def _get_sorted_order(lens):
    sorted_len, fwd_order = torch.sort(
      lens.contiguous().view(-1), 0, descending=True
    )
    _, bwd_order = torch.sort(fwd_order)
    sorted_len = list(sorted_len)
    return sorted_len, fwd_order, bwd_order