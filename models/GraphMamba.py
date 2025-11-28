import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from trainer.Token_Create import precompute_graph_and_tokens


class GraphMambaGMN(nn.Module):
    def __init__(self, configs, args):
        super().__init__()
        d_model = configs.dimension_token
        self.subgraph_encoder = nn.Sequential(
            nn.Linear(configs.hidden_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.local_mamba = nn.ModuleList([
            BidirectionalMamba()
            for _ in range(configs.num_local_layers)
        ])
        self.global_mamba = nn.ModuleList([
            BidirectionalMamba()
            for _ in range(configs.num_global_layers)
        ])
        if configs.use_mpnn:
            self.mpnn = MPNN_nk(d_model, d_model)
        else:
            self.mpnn = None

        self.num_nodes = None
        self.adj_list = None
        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.perm = None  # 度排序
        self.inv_perm = None


    def _encode_subgraph_tokens_batched(self, x):
        b_samples, num_nodes, _ = x.size()
        device = x.device
        l_token = len(self.tokens_per_node[0])
        node_token_feats = torch.zeros(b_samples, num_nodes, l_token, self.d_model, device=device)
        for v in range(num_nodes):
            tokens_v = self.tokens_per_node[v]
            for t, node_ids in enumerate(tokens_v):
                idx = torch.tensor(node_ids, dtype=torch.long, device=device)
                sub_x = x[:, idx, :] # (k, in_dim)
                pooled = sub_x.mean(dim=1) # (in_dim)
                node_token_feats[:, v, t, :] = self.subgraph_encoder(pooled)
        return node_token_feats



    def forward(self, x, adj):
        device = x.device
        if self.tokens_per_node is None:
            self.num_nodes, self.adj_list, self.edge_index, self.tokens_per_node, self.perm, self.inv_perm = precompute_graph_and_tokens(adj)
        b_samples, num_nodes, _ = x.size()
        perm = self.perm.to(device)
        inv_perm = self.inv_perm.to(device)
        node_token_feats = self._encode_subgraph_tokens_batched(x)
        b_samples, num_nodes, time_length, dimension = node_token_feats.size()
        h = node_token_feats(b_samples * num_nodes, time_length, dimension)
        for layer in self.local_mamba:
            h = layer(h)
        node_repr_all = h[:, -1, :] # (B*N, D)
        node_repr = node_repr_all.view(b_samples, num_nodes, dimension)
        node_seq_sorted = node_repr[:, perm, :]
        h_global = node_seq_sorted
        for layer in self.global_mamba:
            h_global = layer(h_global)

        node_repr_global = h_global[:, inv_perm, :] # (B, N, D)
        if self.use_mpnn and self.mpnn is not None:
            edge_index = self.edge_index.to(device)
            mpnn_out = self.mpnn(node_repr_global, edge_index)
            node_repr_global = node_repr_global + F.relu(mpnn_out)

        return node_repr_global





class BidirectionalMamba(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.fwd = Mamba(
            d_model = configs.dimension_token,
            d_state = configs.dimension_state,
            d_conv = configs.dimension_conv,
            expand = configs.expand
        )

        self.bwd = Mamba(
            d_model = configs.dimension_token,
            d_state = configs.dimension_state,
            d_conv = configs.dimension_conv,
            expand = configs.expand
        )
        self.norm = nn.LayerNorm(configs.dimension_token)

    def forward(self, x):
        x_norm = self.norm(x)
        y_fwd = self.fwd(x_norm)
        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd_rev = self.bwd(x_rev)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        y = 0.5 * (y_fwd + y_bwd)
        return y




class MPNN_nk(nn.Module):
    def __init__(self, in_fea, out_fea):
        super().__init__()
        self.lin_self = nn.Linear(in_fea, out_fea)
        self.lin_neig = nn.Linear(in_fea, out_fea)

    def forward(self, curr_fea, edge_index):
        num_nodes, dimension = curr_fea.size()
        device = curr_fea.device()
        src, dst = edge_index
        src_all = torch.cat([src, dst], dim=0)
        dst_all = torch.cat([dst, src], dim=0)
        nei_sum = torch.zeros(num_nodes, dimension, device=device)
        nei_sum.index_add_(0, dst_all, curr_fea[src_all])
        out_fea = self.lin_self(curr_fea) + self.lin_neig(nei_sum)
        return out_fea



class GraphMambaLayer(nn.Module):
    """
    一个简化版的 GraphMamba 图层：
    - 输入：节点特征 x、边 edge_index、batch（节点属于哪个图）
    - 步骤：
      1) 线性映射到 d_model 维度
      2) 逐图计算节点度数，按度数排序成序列
      3) 对每个图 pad 到相同长度 -> [B, L, d_model]
      4) 送入 Mamba 进行序列建模
      5) 把输出序列按原顺序映射回每个节点
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # 节点特征映射到 Mamba 所需维度
        self.in_proj = nn.Linear(in_dim, d_model)

        # Mamba 块：标准一维序列 Mamba
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 输出映射回 out_dim
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x, edge_index, batch):
        """
        x:          [N, in_dim]
        edge_index: [2, E]
        batch:      [N]，每个节点所属图的 id（0 ~ B-1）
        """
        device = x.device
        N = x.size(0)

        # 1) 映射到 d_model 维度
        h = self.in_proj(x)  # [N, d_model]

        # 2) 计算每个节点的度数，用作简单的“重要性”排序依据
        deg = degree(edge_index[0], num_nodes=N, dtype=h.dtype)  # [N]

        # 3) 统计 batch 中图的数量以及每个图的节点数
        num_graphs = int(batch.max().item()) + 1
        node_counts = torch.bincount(batch, minlength=num_graphs)  # [B]
        max_len = int(node_counts.max().item())

        seqs = []   # 存储每个图的序列化节点特征 [max_len, d_model]
        perms = []  # 存储每个图的“序列 -> 原节点索引”映射

        for g in range(num_graphs):
            # 当前图 g 的所有节点索引
            node_mask = (batch == g)
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)  # [n_g]

            # 根据度数排序（从大到小），构造“节点序列”
            scores = deg[node_idx]
            order = torch.argsort(scores, descending=True)
            perm = node_idx[order]  # [n_g]，序列位置 -> 原始节点 index
            perms.append(perm)

            h_g = h[perm]      # [n_g, d_model]
            n_g = h_g.size(0)

            # pad 到 max_len，方便拼成 [B, L, d_model]
            if n_g < max_len:
                pad = h_g.new_zeros(max_len - n_g, h_g.size(1))
                h_g = torch.cat([h_g, pad], dim=0)  # [max_len, d_model]

            seqs.append(h_g.unsqueeze(0))  # [1, max_len, d_model]

        # 4) 拼成 batch 序列，输入 Mamba
        #    形状: [B, max_len, d_model]
        seqs = torch.cat(seqs, dim=0).to(device)
        y_seq = self.mamba(seqs)  # [B, max_len, d_model]

        # 5) 把序列输出映射回每个节点
        out = h.new_zeros(N, self.d_model)  # [N, d_model]

        for g in range(num_graphs):
            n_g = int(node_counts[g].item())
            perm = perms[g]  # [n_g]
            out[perm] = y_seq[g, :n_g, :]  # 把前 n_g 个有效位置写回

        # 6) 最后一个线性层
        return self.out_proj(out)  # [N, out_dim]


class GraphMambaNet(nn.Module):
    """
    一个简单的节点分类网络：
    GCNConv 局部消息传递 + GraphMambaLayer 全局长程建模
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.mamba_layer = GraphMambaLayer(
            in_dim=hidden_channels,
            out_dim=hidden_channels,
            d_model=hidden_channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        # 如果只有一个图，手动造一个 batch 向量
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 两层 GCN 做局部聚合
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # GraphMamba 层做长程建模
        x = self.mamba_layer(x, edge_index, batch)

        # 节点分类：对每个节点输出一个 logits
        out = self.head(x)  # [N, num_classes]
        return out



class BidirectionalMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4):
        super().__init__()
        # 正向 / 反向各一个 Mamba
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.fwd(x)          # (B, L, d_model)

        x_rev = torch.flip(x, dims=[1])
        y_bwd_rev = self.bwd(x_rev)  # (B, L, d_model)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        # 简单平均聚合前向/反向
        return 0.5 * (y_fwd + y_bwd)