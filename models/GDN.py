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
              self.searchable_gnn = SearchableGNNLayer(
                  input_dim=input_dim,
                  model_dim=dim,
                  embed_dim=embed_dim,
                  node_num=node_num,
                  search_space=search_space  # 传递搜索空间
              )
              self.gnn_layers = None
        else:
              self.gnn_layers = nn.ModuleList([
                  GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1),
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

        # =====================================================
        # Case 1: 使用 NAS 搜索模型
        # =====================================================

        if self.searchable_gnn is not None and self.use_nas:
            # ---- 1) 基于节点 embedding 构建 Top-K 图 ----
            node_ids = torch.arange(node_num, device=device)   # [N]
            base_embeddings = self.embedding(node_ids)         # [N, embed_dim]

            # 用 detach 的 embedding 计算相似度，避免梯度回传到 embedding
            weights = base_embeddings.detach()                 # [N, embed_dim]
            cos_ji_mat = weights @ weights.T                   # [N, N]
            norm = weights.norm(dim=-1, keepdim=True)          # [N, 1]
            cos_ji_mat = cos_ji_mat / (norm @ norm.T + 1e-8)   # [N, N]

            topk_num = self.topk
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]  # [N, K]
            self.learned_graph = topk_indices_ji                            # 保留以便可视化

            # i 为被指向节点，j 为指向它的 Top-K 邻居
            gated_i = node_ids.unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0)  # [1, N*K]
            gated_j = topk_indices_ji.flatten().unsqueeze(0)                            # [1, N*K]
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)                     # [2, N*K]

            # 扩展到 batch：第 b 个样本整体平移 b*node_num
            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device).long()                                                         # [2, B*N*K]

            # ---- 2) 准备给 GraphLayer 用的 embedding（[B*N, embed_dim]） ----
            all_embeddings = base_embeddings.repeat(batch_num, 1)   # [B*N, embed_dim]

            # ---- 3) 交给 SearchableGNNLayer ----
            gcn_out = self.searchable_gnn(
                x,                          # [B*N, F]
                batch_gated_edge_index,     # 稀疏 Top-K 图
                embedding=all_embeddings,   # GraphLayer 使用
                node_num=batch_num * node_num
            )

            x = gcn_out.view(batch_num, node_num, -1)   # [B, N, C']

        # =====================================================
        # Case 2: 使用原始 GDN 模式（非 NAS）
        # =====================================================
        else:
            gcn_outs = []
            for i, edge_index in enumerate(edge_index_sets):
                edge_num = edge_index.shape[1]
                cache_edge_index = self.cache_edge_index_sets[i]

                if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                    self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

                batch_edge_index = self.cache_edge_index_sets[i]

                #  self.embedding 获取每个节点的嵌入向量
                all_embeddings = self.embedding(torch.arange(node_num).to(device))

                weights_arr = all_embeddings.detach().clone()
                all_embeddings = all_embeddings.repeat(batch_num, 1)

                weights = weights_arr.view(node_num, -1)

                # 计算节点间余弦相似度
                cos_ji_mat = torch.matmul(weights, weights.T)
                normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
                cos_ji_mat = cos_ji_mat / normed_mat

                dim = weights.shape[-1]
                topk_num = self.topk

                #  选择Top-K相似节点
                # dim=-1 表示在最后一个维度（即每个节点的相似度向量）上找top-k
                # torch.topk 函数返回一个包含两个元素的元组：
                # 第一个元素 [0]：top-k 的值（values）
                # 第二个元素 [1]：top-k 的索引（indices）
                # [1] 表示只取 torch.topk 返回结果中的索引部分，而不是值部分
                topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

                self.learned_graph = topk_indices_ji

                gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
                gated_j = topk_indices_ji.flatten().unsqueeze(0)
                # 构建新的图连接关系 gated_edge_index
                gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

                # 把单批次的图结构扩展到整个批次，得到整个批次的边索引
                batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
                # 调用第 i 个 GNNLayer 进行图卷积操作
                gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

                gcn_outs.append(gcn_out)

            x = torch.cat(gcn_outs, dim=1)
            x = x.view(batch_num, node_num, -1)

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


    def get_search_space(self):
        """
        返回可搜索的架构空间，只有在启用 NAS 时才会存在
        """
        if self.searchable_gnn:
            return self.searchable_gnn.search_space  # 返回可搜索的空间
        else:
            # 如果没有启用 NAS，返回 None 或空字典
            return None