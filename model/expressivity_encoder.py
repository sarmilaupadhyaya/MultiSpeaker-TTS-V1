import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out

class crnn_self_attention_pool_encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(crnn_self_attention_pool_encoder, self).__init__()
        
        self.input_dim = input_dim
        self.conv = nn.Conv2d(1,1,3,1,1)
        self.rnn = nn.LSTM(input_dim, 256, 1, bidirectional=True, batch_first=True)
        self.FF = nn.Linear(1024, 512)
        self.last_layer = nn.Linear(512, output_dim)
        self.attention = MultiHeadAttention(512,512,512,1)
        self.selu = nn.SELU()
        
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,2) - torch.mul(mean,mean)
        return variance
        
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        
        mean = torch.mean(el_mat_prod,2)
        
        variance = self.weighted_sd(inputs,attention_weights,mean)
        
        #print(mean.shape, variance.shape)
        
        stat_pooling = torch.cat((mean,variance),1)
        
        return stat_pooling
        
    def forward(self, x):
        #print(x.shape)        
        x = x.unsqueeze(1)
        o = self.conv(x)
        o = o.squeeze(1)
        o,(h,c) = self.rnn(o)
        h = h.permute(1,0,2)
        h = h.reshape(h.shape[0],h.shape[1]*h.shape[2]).unsqueeze(1)        
        attn_weights = self.attention(h,o).squeeze(1)
        o = self.stat_attn_pool(o.permute(0,2,1),attn_weights)
        o = self.FF(o)
        o = self.selu(o)
        o = self.last_layer(o).unsqueeze(-1)
                
        return o


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EmbeddingNetwork, self).__init__()

        self.out_dim = out_dim
        self.in_dim = in_dim
        
        self.encoder = crnn_self_attention_pool_encoder(in_dim,out_dim)
        
    def forward(self, x):
        o = self.encoder(x)              
        
        return o

