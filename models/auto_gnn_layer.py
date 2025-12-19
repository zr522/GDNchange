# models/auto_gnn_layer.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE,GraphConv,GENConv

# 可选：你的 GraphLayer（与注意力/embedding耦合的那层）
from .graph_layer import GraphLayer  # 已存在文件:contentReference[oaicite:5]{index=5}

# auto_gnn_layer.py 顶部（import 后面随便放）
class EmbeddingGNN(nn.Module):
    """
    约定接口:
      forward(x, edge_index, embedding=None, return_attention_weights=False)
    用来标记“会用到 embedding 的 GNN 层”，方便在 SearchableGNNLayer 里统一处理。
    """
    def forward(self, x, edge_index, embedding=None, return_attention_weights=False):
        raise NotImplementedError


class EmbeddingAwareConv(nn.Module):
    """
    约定：forward(x, edge_index, embedding=None)
    - x: [B*N, Fin]
    - edge_index: [2, E*B]
    - embedding: [B*N, embed_dim] 或 None
    """
    def forward(self, x, edge_index, embedding=None):
        raise NotImplementedError

class GraphConvEmb(EmbeddingAwareConv):
    """
    GraphConv + sensor embedding:
      x' = GraphConv(x + W_e * emb)
    """
    def __init__(self, in_dim, out_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, in_dim, bias=False)
        self.conv = GraphConv(in_dim, out_dim)

    def forward(self, x, edge_index, embedding=None):
        if embedding is not None:
            # embedding: [B*N, embed_dim]
            x = x + self.proj(embedding)
        return self.conv(x, edge_index)

class GENConvEmb(EmbeddingAwareConv):
    """
    GENConv + sensor embedding:
      x' = GENConv(x + W_e * emb)
    这里给一个相对稳的配置，适合小图：
      aggr='softmax', t=1.0, learn_t=True
    """
    def __init__(self, in_dim, out_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, in_dim, bias=False)
        self.conv = GENConv(
            in_dim,
            out_dim,
            aggr='softmax',     # 比 mean 更有“注意力味道”
            t=1.0,
            learn_t=True,
            num_layers=2,       # 内部小MLP层数
        )

    def forward(self, x, edge_index, embedding=None):
        if embedding is not None:
            x = x + self.proj(embedding)
        return self.conv(x, edge_index)

class GCNWithEmb(EmbeddingAwareConv):
    def __init__(self, in_dim, out_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, in_dim, bias=False)
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, embedding=None):
        if embedding is not None:
            # embedding: [B*N, embed_dim]
            emb_proj = self.proj(embedding)
            x = x + emb_proj
        return self.conv(x, edge_index)


class GATWithEmb(EmbeddingAwareConv):
    def __init__(self, in_dim, out_dim, embed_dim, heads=1):
        super().__init__()
        self.proj = nn.Linear(embed_dim, in_dim, bias=False)
        # concat=False 保证输出维度仍为 out_dim
        self.conv = GATConv(in_dim, out_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, embedding=None):
        if embedding is not None:
            emb_proj = self.proj(embedding)
            x = x + emb_proj
        return self.conv(x, edge_index)


class SAGEWithEmb(EmbeddingAwareConv):
    def __init__(self, in_dim, out_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, in_dim, bias=False)
        self.conv = GraphSAGE(in_dim, out_dim,num_layers=2)

    def forward(self, x, edge_index, embedding=None):
        if embedding is not None:
            emb_proj = self.proj(embedding)
            x = x + emb_proj
        return self.conv(x, edge_index)

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
                # 图层深度：1=原始GDN，2~3为轻量加深
                "depth": [2, 3,4,5],
                # 候选操作：全部都是“看得见 embedding 的 GNN”
                "op_candidates": [
                    "GraphLayer",      # baseline：单头 embedding 注意力
                    "GraphLayerMH",    # 多头 embedding 注意力（强调多关系模式）
                    "GATEmb",          # 特征+embedding 融合后做GAT
                    "SAGEEmb",         # 特征+embedding 融合后做GraphSAGE
                    "GCNEmb",          # 特征+embedding 融合后做GCN（更平滑、稳一点）
                    "GraphConvEmb",
                    "GENConvEmb",
                ],

                # 每层隐藏维度：以64为中心做小范围搜索，避免太大过拟合
                "hidden_candidates": [32, 64, 128,256],

                # 跳跃连接：1层时一般就 none；2~3层时 residual 能稳住训练
                "skip_candidates": ["none", "residual","gated"],
            }

        self.search_space = search_space

        # 运行时构建的架构
        self.depth = 1
        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hidden_dims = []
        self.skip = "none"
        self.skip_gates = nn.ParameterList()  # 新增：为 "gated" 准备门控参数列表
        # 新增：每一层一个 embedding -> in_dim 的映射
        self.embed_fcs = nn.ModuleList()

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
             "skip": "none"|"residual"|"gated"
          }
        """
        self.depth = int(arch_config["depth"])
        assert self.depth >= 1
        assert len(arch_config["ops"]) == self.depth, f"Length of ops ({len(arch_config['ops'])}) must match depth ({self.depth})"
        assert len(arch_config["hidden_dims"]) == self.depth, f"Length of hidden_dims ({len(arch_config['hidden_dims'])}) must match depth ({self.depth})"


        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_gates = nn.ParameterList()   # 重新初始化门控列表

        self.embed_fcs = nn.ModuleList()       # 👉 重新初始化 embedding 融合层

        self.hidden_dims = list(arch_config["hidden_dims"])
        self.skip = arch_config.get("skip", "none")

        in_dim = self.input_dim
        for i in range(self.depth):
            if i >= len(arch_config["ops"]):
                raise IndexError(f"Index {i} out of range for ops list (length {len(arch_config['ops'])})")

            op_type = arch_config["ops"][i]
            out_dim = self.hidden_dims[i]

            # 记录当前层卷积前的输入维度，用于构造 embed_fc
            current_in_dim = in_dim

            if op_type == "GraphLayer":
                # 原始 GDN 行为（无异常感知项）
                conv = GraphLayer(
                    in_dim,
                    out_dim,
                    heads=1,
                    concat=False,
                    inter_dim=out_dim + self.embed_dim,
                )
            # 注意：GraphLayer.forward(x, edge_index, embedding, return_attention_weights=False)
            elif op_type == "GraphLayerMH":
                    # 多头版本（例如4头），concat=False 保证输出维度仍为 out_dim
                    conv = GraphLayer(
                        in_dim,
                        out_dim,
                        heads=4,
                        concat=False,
                        inter_dim=out_dim + self.embed_dim,
                    )
            elif op_type == "GCNEmb":
                conv = GCNWithEmb(in_dim, out_dim, self.embed_dim)

            elif op_type == "GATEmb":
                conv = GATWithEmb(in_dim, out_dim, self.embed_dim, heads=2)  # 也可设为4，看显存

            elif op_type == "SAGEEmb" or op_type == "GraphSAGEEmb":
                conv = SAGEWithEmb(in_dim, out_dim, self.embed_dim)

            elif op_type == "GraphConvEmb":
                conv = GraphConvEmb(in_dim, out_dim, self.embed_dim)

            elif op_type == "GENConvEmb":
                conv = GENConvEmb(in_dim, out_dim, self.embed_dim)
            else:
                raise ValueError(f"Unsupported op_type: {op_type}")

            self.ops.append(conv)
            self.bns.append(nn.BatchNorm1d(out_dim))  # [N, C] 形式
            # 为每一层准备一个标量门控参数（只在 skip=="gated" 时真正发挥作用）
            self.skip_gates.append(nn.Parameter(torch.zeros(1)))
            # 新增：这一层的 embedding -> 当前层输入维度 的线性变换
            self.embed_fcs.append(
                nn.Linear(self.embed_dim, current_in_dim, bias=False)
            )
            in_dim = out_dim

        self._ensure_final_proj()
        # 简单的激活函数
        self.relu = nn.ReLU()

        # 轻量dropout，避免搜索阶段过拟合
        self.dp = nn.Dropout(0.1)

    def forward(self, x, edge_index, embedding=None, node_num=0):
        """
        输入:
          x:         [B*N, Fin]
          edge_index:[2, E*B]
          embedding: [B*N, embed_dim]，来自 GDN 里的 all_embeddings
          node_num:  B*N（现在其实用不到，只是保持接口兼容）
        输出:
          out: [B*N, model_dim]
        """
        device = x.device
        edge_index = edge_index.long().to(device)
        out = x
        prev = None

        for i, conv in enumerate(self.ops):
            # -------- 1) 先做一层基于 embedding 的线性注入 --------
            h = out
            if embedding is not None:
                # embedding: [B*N, embed_dim]
                # embed_fc_i: embed_dim -> 当前层输入维度 in_dim
                emb_proj = self.embed_fcs[i](embedding.to(device))  # [B*N, in_dim]
                h = h + emb_proj

            # -------- 2) 根据算子类型决定是否把 embedding 传进去 --------
            if isinstance(conv, GraphLayer) or isinstance(conv, EmbeddingAwareConv):
                # GraphLayer / *Emb 系列：接受 (h, edge_index, embedding)
                emb = embedding.to(device) if embedding is not None else None
                out = conv(h, edge_index, emb)
            else:
                # 其它 PyG 原生算子：只吃 (h, edge_index)
                out = conv(h, edge_index)

            # -------- 3) BN + ReLU --------
            out = self.bns[i](out)   # BatchNorm1d: 输入 [B*N, C]
            out = self.relu(out)

            # -------- 4) 残差 / 门控残差 --------
            if prev is not None and prev.shape == out.shape:
                if self.skip == "residual":
                    out = out + prev
                elif self.skip == "gated":
                    gate = torch.sigmoid(self.skip_gates[i])  # 标量门控
                    out = gate * out + (1.0 - gate) * prev
                # 'none' 就什么都不做

            prev = out
            out = self.dp(out)  # Dropout

        # -------- 5) 统一投影到 model_dim --------
        if self.final_proj is not None:
            out = self.final_proj(out)

        return out
