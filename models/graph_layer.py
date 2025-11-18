import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add',node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # 推断 embedding 维度：假设 inter_dim = out_channels + embed_dim（如果被正确传入）
        if inter_dim is not None and inter_dim > 0:
            self.embed_dim = inter_dim - out_channels
        else:
            # 回退：如果没传 inter_dim，就假设 embed_dim == out_channels（兼容原始 GDN）
            self.embed_dim = out_channels

        # 特征部分和 embedding 部分各自的注意力参数
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, self.embed_dim))
        self.att_em_j = Parameter(torch.Tensor(1, heads, self.embed_dim))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)



    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        device = x.device  # 获取输入的设备
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            # 拼接节点特征和embedding
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

            # 注意力参数同样拼接两部分
            cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
            cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

            # ----------- CASE 2: embedding 为空 -----------
        else:
            key_i = x_i
            key_j = x_j

            # 注意力参数只使用特征部分
            cat_att_i = self.att_i
            cat_att_j = self.att_j

            # ---- 计算注意力系数 ----
        assert key_i.shape[-1] == cat_att_i.shape[-1], \
            f"Shape mismatch after fix: {key_i.shape[-1]} vs {cat_att_i.shape[-1]}"

        if key_i.shape[-1] != cat_att_i.shape[-1]:
            # 调整大小，使它们匹配
            # 可以选择做扩展或者调整尺寸
            cat_att_i = cat_att_i.expand(-1, -1, key_i.shape[-1])


        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)


        alpha = alpha.view(-1, self.heads, 1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)



    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
