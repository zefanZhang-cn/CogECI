from torch import nn
import torch.nn.functional as F
import torch
import copy
import math
from torch.autograd import Variable
class GraphConvLayer(nn.Module):
    """A GCN module operated on dependency graphs."""

    def __init__(self, dropout, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(dropout)
        self.fc_k = nn.Linear(mem_dim, mem_dim)
        self.fc_q = nn.Linear(mem_dim, mem_dim)
        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def build_graph(self, x):
        batch_size, s_len = x.shape[0], x.shape[1]
        emb_k = self.fc_k(x)
        emb_q = self.fc_q(x)
        length = torch.tensor([s_len] * batch_size, dtype=torch.long)

        s = torch.bmm(emb_k, emb_q.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        for i, (l_1, l_2) in enumerate(zip(length, length)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))

        A = s  # F.softmax(s, dim=2)  # [B, t1, t2]

        A.data.masked_fill_(A.data != A.data, 0)

        return A

    def forward(self, adj, gcn_inputs):

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """A GCN module operated on dependency graphs."""

    def __init__(self, dropout, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):
        gcn_inputs = gcn_inputs.sum(2).squeeze(2)# [16,23,768]
        adj_list = adj_list.unsqueeze(0).unsqueeze(-1).expand(4, 16, 23, 23)# 4,16,23,23
        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(1).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn