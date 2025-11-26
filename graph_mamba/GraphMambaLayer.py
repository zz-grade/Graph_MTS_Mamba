import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch_geometric.utils import degree


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

