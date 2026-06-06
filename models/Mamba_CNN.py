import torch
import torch.nn as nn


class NodeTemporalConv(nn.Module):
    """
    对每个节点分别执行 MambaSL 风格的时间卷积。

    输入:  [B, L*N, H]
    输出:  [B, N, L, D]
    """
    def __init__(
        self,
        hidden_channels,
        d_model,
        num_nodes,
        kernel_size=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes

        self.token_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )

        nn.init.kaiming_normal_(
            self.token_conv.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    def forward(self, x):
        # x: [B, L*N, H]
        B, total_nodes, H = x.shape
        N = self.num_nodes

        if total_nodes % N != 0:
            raise ValueError(
                f"total_nodes={total_nodes} cannot be divided by N={N}"
            )

        L = total_nodes // N

        # 当前展平顺序为时间优先：
        # [B,L*N,H] -> [B,L,N,H]
        x = x.reshape(B, L, N, H)

        # 每个节点形成一条时间序列
        # [B,L,N,H] -> [B,N,H,L] -> [B*N,H,L]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(B * N, H, L)

        # [B*N,H,L] -> [B*N,D,L]
        x = self.token_conv(x)

        # [B*N,D,L] -> [B,N,L,D]
        D = x.size(1)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(B, N, L, D)

        return x