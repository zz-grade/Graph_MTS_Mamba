import torch.nn as nn
from torch_geometric.nn import GINEConv

class DiffPoolTokenAggregator(nn.Module):
    """
        DiffPool-like soft assignment:
          - 用 GNN 生成每个节点到 K 个中心的分配 logits
          - softmax 得到 S (B,N,K)
          - 用 X_pool = S^T * X 聚合得到 (B,K,D)
        注意：这里不算 A_pool = S^T A S（省显存），只做 token 压缩给 Mamba。
        """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            k_centers: int = 32,
            num_gnn_layers: int = 2,
            dropout: float = 0.0,
            use_layernorm: bool = True,
            tau: float = 1.0,  # softmax temperature，<1 更“硬”
    ):
        super().__init__()

        self.k = k_centers
        self.dropout = dropout
        self.tau = tau

        # 1) embedding GNN：把输入特征映射到 out_dim（用于后续聚合）
        self.embed_in = nn.Linear(in_dim, hidden_dim)

        self.embed_convs = nn.ModuleList()
        self.embed_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.embed_convs.append(GINEConv(mlp))
            self.embed_norms.append(nn.LayerNorm(hidden_dim))

        self.embed_out = nn.Linear(hidden_dim, out_dim)

        # 2) assignment GNN：输出 K 维 logits
        self.assign_convs = nn.ModuleList()
        self.assign_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.assign_convs.append(GINEConv(mlp))
            self.assign_norms.append(nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity())

        self.assign_out = nn.Linear(hidden_dim, k_centers)