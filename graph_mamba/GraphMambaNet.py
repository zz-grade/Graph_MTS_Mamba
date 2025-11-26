from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F

from graph_mamba.GraphMambaLayer import GraphMambaLayer


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
