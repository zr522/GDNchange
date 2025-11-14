# models/auto_gnn_layer.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# 可选：你的 GraphLayer（与注意力/embedding耦合的那层）
from .graph_layer import GraphLayer  # 已存在文件:contentReference[oaicite:5]{index=5}

class SearchableGNNLayer(nn.Module):
    """
    一个可搜索的GNN组合层，保持与原GNNLayer一致的forward签名：
      forward(x, edge_index, embedding=None, node_num=0) -> Tensor

    设计：
      - depth ∈ {1, 2, 3}
      - 每层 op ∈ {GraphLayer, GCN, GAT, SAGE}
      - 每层 hidden_dim ∈ {32, 64, 128}
      - skip ∈ {none, residual}
      - 最后统一投影到 model_dim (即 GDN.__init__ 传入的 dim)，从而保持 OutLayer 输入维度不变:contentReference[oaicite:6]{index=6}
    """
    def __init__(
            self,
            input_dim: int,
            model_dim: int,
            embed_dim: int,
            node_num: int,
            search_space: dict = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim         # 最终投影到的维度（与GDN中的dim一致）
        self.embed_dim = embed_dim         # 节点embedding维度
        self.node_num = node_num

        # 搜索空间（可按需扩展/裁剪）
        if search_space is None:
            search_space = {
                "depth": [1, 2, 3, 4, 5],  # 增加网络深度
                "op_candidates": ["GraphLayer", "GCN", "GAT", "SAGE", "GraphSAGE"],  # 增加GNN操作候选项
                "hidden_candidates": [32, 64, 128],  # 增加隐藏层维度候选值
                "skip_candidates": ["none", "residual", "dense"],  # 增加更多跳跃连接候选项
            }
        self.search_space = search_space

        # 运行时构建的架构
        self.depth = 1
        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hidden_dims = []
        self.skip = "none"

        # 默认先构建一套可用的结构，避免未build时forward报错
        self.build_arch({
            "depth": 1,
            "ops": ["GraphLayer"],
            "hidden_dims": [self.model_dim],
            "skip": "none",
        })

        # 若最后一层输出维度 != model_dim，则使用线性投影
        self.final_proj = None
        self._ensure_final_proj()

    def get_search_space(self):
        return self.search_space

    def _ensure_final_proj(self):
        last_dim = self.hidden_dims[-1]
        if last_dim != self.model_dim:
            self.final_proj = nn.Linear(last_dim, self.model_dim)
        else:
            self.final_proj = None

    def build_arch(self, arch_config: dict):
        """
        根据 arch_config 创建实际的层堆叠：
          arch_config = {
             "depth": int,
             "ops":   [str]*depth,       # 每层的op
             "hidden_dims": [int]*depth, # 每层的输出维度
             "skip": "none"|"residual"
          }
        """
        self.depth = int(arch_config["depth"])
        assert self.depth >= 1

        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hidden_dims = list(arch_config["hidden_dims"])
        self.skip = arch_config.get("skip", "none")

        in_dim = self.input_dim
        for i in range(self.depth):
            op_type = arch_config["ops"][i]
            out_dim = self.hidden_dims[i]

            if op_type == "GraphLayer":
                # 你的 GraphLayer 含 embedding 注意力项:contentReference[oaicite:7]{index=7}
                conv = GraphLayer(
                    in_dim,
                    out_dim,
                    heads=1,
                    concat=False,
                    inter_dim=out_dim + self.embed_dim
                )
                # 注意：GraphLayer.forward(x, edge_index, embedding, return_attention_weights=False)
            elif op_type == "GCN":
                conv = GCNConv(in_dim, out_dim)
            elif op_type == "GAT":
                conv = GATConv(in_dim, out_dim, heads=1)
            elif op_type == "SAGE":
                conv = SAGEConv(in_dim, out_dim)
            elif op_type == "GraphSAGE":  # 确保支持 GraphSAGE
                conv = SAGEConv(in_dim, out_dim)  # GraphSAGE 通常使用 SAGEConv
            else:
                raise ValueError(f"Unsupported op_type: {op_type}")

            self.ops.append(conv)
            self.bns.append(nn.BatchNorm1d(out_dim))  # [N, C] 形式
            in_dim = out_dim

        self._ensure_final_proj()
        # 简单的激活函数
        self.relu = nn.ReLU()

        # 轻量dropout，避免搜索阶段过拟合
        self.dp = nn.Dropout(0.1)

    def forward(self, x, edge_index, embedding=None, node_num=0):
        """
        输入:
          x: [B*N, Fin]
          edge_index: [2, E*B] 已经做了batch展开:contentReference[oaicite:8]{index=8}
          embedding: [N, embed_dim] 被GraphLayer使用，其它算子忽略
        输出:
          out: [B*N, model_dim]
        """
        device = x.device  # 获取输入张量的设备，确保所有其他张量与其一致
        edge_index = edge_index.long().to(device)
        out = x
        prev = None

        for i, conv in enumerate(self.ops):
            if isinstance(conv, GraphLayer):
                # GraphLayer 支持 embedding 参数:contentReference[oaicite:9]{index=9}
                embedding = embedding.to(device) if embedding is not None else None
                out = conv(out, edge_index, embedding)
            else:
                # 其它PyG算子忽略 embedding
                out = conv(out, edge_index)

            # BN + ReLU（BatchNorm1d 期望 [N, C]）
            out = self.bns[i](out)
            out = self.relu(out)

            if self.skip == "residual" and prev is not None and prev.shape == out.shape:
                out = out + prev
            prev = out
            out = self.dp(out)

        # 统一投影到 model_dim
        if self.final_proj is not None:
            out = self.final_proj(out)

        return out
