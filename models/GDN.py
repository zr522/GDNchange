import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer
from .auto_gnn_layer import SearchableGNNLayer   # 新增



# def get_batch_edge_index(org_edge_index, batch_num, node_num):
#     # org_edge_index:(2, edge_num)
#     edge_index = org_edge_index.clone().detach()
#     edge_num = org_edge_index.shape[1]
#     batch_edge_index = edge_index.repeat(1,batch_num).contiguous()
#
#     for i in range(batch_num):
#         batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num
#
#     return batch_edge_index.long()


# models/GDN.py

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # 确保 edge_index 是二维的
    edge_index = org_edge_index.clone().detach()

    if edge_index.dim() != 2:
        raise ValueError(f"Expected edge_index to be 2D, but got {edge_index.dim()}D")

    edge_num = edge_index.shape[1]  # 边的数量 E * B
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()  # 扩展边的数量，复制到 batch_size

    # 为每个批次的边添加偏移量
    for i in range(1, batch_num):
        batch_edge_index[:, i * edge_num: (i + 1) * edge_num] += i * node_num  # 每个批次节点加上偏移量

    return batch_edge_index.long()  # 返回新的边索引




class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    # def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20,MSConv=None,
                                   use_nas: bool = False, search_space: dict = None):
        super(GDN, self).__init__()

        self.use_nas = use_nas  # 添加这一行

        self.MSConv = MSConv

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)

        # self.gnn_layers = nn.ModuleList([
        #     GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1),
        # ])

        if use_nas:
            # 保证输出维度仍为 dim，方便后续 OutLayer
            #   self.searchable_gnn = SearchableGNNLayer(
            #       input_dim=input_dim,
            #       model_dim=dim,
            #       embed_dim=embed_dim,
            #       node_num=node_num,
            #       search_space=search_space  # 传递搜索空间
            #   )
            #   self.gnn_layers = None
            self.searchable_gnn_layers = nn.ModuleList([
                SearchableGNNLayer(
                    input_dim=input_dim,
                    model_dim=dim,         # 每个分支仍然输出 dim
                    embed_dim=embed_dim,
                    node_num=node_num,
                    search_space=search_space,
                )
                for _ in range(edge_set_num)
            ])
            self.gnn_layers = None
        else:
              self.gnn_layers = nn.ModuleList([
                  GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1)
              ])
              self.searchable_gnn = None  # 没有使用


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, org_edge_index):
        x = data.clone().detach()
        device = data.device
        # 引入时序卷积
        if self.MSConv is not None:
            x = self.MSConv(x)

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        edge_index_sets = self.edge_index_sets

        # print(f"batch_edge_index shape: {batch_edge_index.shape}")

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            batch_edge_index = self.cache_edge_index_sets[i]

            # 下面这块保持和原始 GDN 一致：用 embedding 做 Top-K 图
            base_embeddings = self.embedding(torch.arange(node_num, device=device))  # [N, embed_dim]
            weights = base_embeddings.detach()
            cos_ji_mat = weights @ weights.T
            norm = weights.norm(dim=-1, keepdim=True)
            cos_ji_mat = cos_ji_mat / (norm @ norm.T + 1e-8)

            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]          # [N, K]
            self.learned_graph = topk_indices_ji
            gated_i = torch.arange(0, node_num, device=device).unsqueeze(1) \
                .repeat(1, self.topk).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)                 # [2, N*K]

            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)

            all_embeddings = base_embeddings.repeat(batch_num, 1)                    # [B*N, embed_dim]

            # 关键分支：这里决定用谁
            if self.searchable_gnn_layers is not None:
                # NAS 路径
                gcn_out = self.searchable_gnn_layers[i](
                    x,
                    batch_gated_edge_index,
                    embedding=all_embeddings,
                    node_num=batch_num * node_num,
                )
            else:
                # 原始 GDN 路径
                gcn_out = self.gnn_layers[i](
                    x,
                    batch_gated_edge_index,
                    node_num=batch_num * node_num,
                    embedding=all_embeddings,
                )

            gcn_outs.append(gcn_out)

        # 多分支输出拼接 → 和原始 GDN 一致
        x = torch.cat(gcn_outs, dim=1)               # [B*N, dim * edge_set_num]
        x = x.view(batch_num, node_num, -1)          # [B, N, dim * edge_set_num]

    # =====================================================
        # Case 1 & Case 2 通用的输出层逻辑
        # =====================================================
        x = torch.mul(x, self.embedding(torch.arange(0, node_num).to(device)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn_outlayer_in(x))
        x = x.permute(0, 2, 1)

        x = self.dp(x)
        out = self.out_layer(x)
        out = out.view(-1, node_num)
        return out


    # def get_search_space(self):
    #     """
    #     返回可搜索的架构空间，只有在启用 NAS 时才会存在
    #     """
    #     if self.searchable_gnn:
    #         return self.searchable_gnn.search_space  # 返回可搜索的空间
    #     else:
    #         # 如果没有启用 NAS，返回 None 或空字典
    #         return None
    def get_search_space(self):
        if hasattr(self, "searchable_gnn_layers") and self.searchable_gnn_layers:
            return self.searchable_gnn_layers[0].search_space
        return None
