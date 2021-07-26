"""Implementation of the deep relational architecture used in https://arxiv.org/pdf/1806.01830.pdf.
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim
import torch.autograd
from collections import OrderedDict

class AttentionHead(nn.Module):

    def __init__(self, n_elems, elem_size, emb_size):
        super(AttentionHead, self).__init__()
        self.sqrt_emb_size = int(math.sqrt(emb_size))
        #queries, keys, values
        self.query = nn.Linear(elem_size, emb_size)
        self.key = nn.Linear(elem_size, emb_size)
        self.value = nn.Linear(elem_size, elem_size)
        # In the paper the authors normalize the projected Q,K and V with layer normalization.
        # 0,1-normalize every projected entity and apply separate gain and bias to each entry in the
        # embeddings. Weights are shared across entites, but not accross Q,K,V or heads.
        self.qln = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.kln = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.vln = nn.LayerNorm(elem_size, elementwise_affine=True)

    def forward(self, x):
        Q = self.qln(self.query(x))
        K = self.kln(self.key(x))
        V = self.vln(self.value(x))
        # softmax is taken over last dimension (rows) of QK': All the attentional weights going into a column/entity
        # of V thus sum up to 1.
        softmax = F.softmax(torch.bmm(Q,K.transpose(1,2))/self.sqrt_emb_size, dim=-1)
        return torch.bmm(softmax,V)

    def attention_weights(self, x):
        Q = self.qln(self.query(x))
        K = self.kln(self.key(x))
        V = self.vln(self.value(x))
        # softmax is taken over last dimension (rows) of QK': All the attentional weights going into a column/entity
        # of V thus sum up to 1.
        softmax = F.softmax(torch.bmm(Q,K.transpose(1,2))/self.sqrt_emb_size, dim=-1)
        return softmax

class AttentionModule(nn.Module):

    def __init__(self, n_elems, elem_size, emb_size, n_heads):
        super(AttentionModule, self).__init__()

        self.heads =  nn.ModuleList(AttentionHead(n_elems, elem_size, emb_size) for _ in range(n_heads))
        self.linear1 = nn.Linear(n_heads*elem_size, elem_size)
        self.linear2 = nn.Linear(elem_size, elem_size)

        self.ln = nn.LayerNorm(elem_size, elementwise_affine=True)

    def forward(self, x):
        #concatenate all heads' outputs
        A_cat = torch.cat([head(x) for head in self.heads], -1)
        # projecting down to original element size with 2-layer MLP, layer size = entity size
        mlp_out = F.relu(self.linear2(F.relu(self.linear1(A_cat))))
        # residual connection and final layer normalization
        return self.ln(x + mlp_out)

    def get_att_weights(self, x):
        #Version of forward function that also returns softmax-normalied QK' attention weights
        #concatenate all heads' outputs
        A_cat = torch.cat([head(x) for head in self.heads], -1)
        # projecting down to original element size with 2-layer MLP, layer size = entity size
        mlp_out = F.relu(self.linear2(F.relu(self.linear1(A_cat))))
        # residual connection and final layer normalization
        output = self.ln(x + mlp_out)
        attention_weights = [head.attention_weights(x).detach() for head in self.heads]
        return [output, attention_weights]
