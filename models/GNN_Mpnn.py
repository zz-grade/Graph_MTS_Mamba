import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import MessagePassing



class MPNNBranch(MessagePassing):
    """一个独立的固定深度 MPNN 分支。"""

    def __init__(self, in_fea, out_fea, depth):
        super().__init__(aggr="add")

        layer_dims = [in_fea] + [out_fea] * (depth - 1)

        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, out_fea),
            )
            for dim in layer_dims
        ])

        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2 + 1, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, 1),
            )
            for dim in layer_dims
        ])

        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + out_fea, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, out_fea),
            )
            for dim in layer_dims
        ])

        self.res_projs = nn.ModuleList([
            nn.Linear(dim, out_fea) if dim != out_fea else nn.Identity()
            for dim in layer_dims
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(out_fea)
            for _ in layer_dims
        ])

    def forward(self, x, edge_index, edge_weight):
        for msg_mlp, gate_mlp, upd_mlp, res_proj, norm in zip(
            self.msg_mlps,
            self.gate_mlps,
            self.upd_mlps,
            self.res_projs,
            self.norms,
        ):
            identity = res_proj(x)

            message = self.propagate(
                edge_index,
                x=x,
                edge_weight=edge_weight,
                msg_mlp=msg_mlp,
                gate_mlp=gate_mlp,
            )

            update = upd_mlp(torch.cat([x, message], dim=-1))
            x = update + identity

        return x

    def message(self, x_i, x_j, edge_weight, msg_mlp, gate_mlp):
        edge_weight = edge_weight.reshape(-1, 1).to(x_j.dtype)

        gate_input = torch.cat(
            [x_i, x_j, edge_weight],
            dim=-1,
        )

        gate = torch.sigmoid(gate_mlp(gate_input))
        message = msg_mlp(x_j)

        return gate * message


class MPNN_some(nn.Module):
    def __init__(
        self,
        in_fea,
        out_fea,
        depths=(1, 3, 5, 7, 9),
    ):
        super().__init__()

        self.depths = tuple(depths)

        # 五个分支完全使用独立参数
        self.branches = nn.ModuleList([
            MPNNBranch(in_fea, out_fea, depth)
            for depth in self.depths
        ])

        # [分支数, 特征维度]
        # 每个特征维度分别学习五种深度的融合比例
        self.fusion_logits = nn.Parameter(torch.zeros(5))

        self.fusion_norm = nn.LayerNorm(out_fea)

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
    ):
        branch_outputs = [
            branch(x, edge_index, edge_weight)
            for branch in self.branches
        ]

        # [num_nodes, 5, out_fea]
        stacked_outputs = torch.stack(branch_outputs, dim=1)
        weights = torch.softmax(self.fusion_logits, dim=0)

        fused_output = torch.sum(
            stacked_outputs * weights.view(1, 5, 1),
            dim=1,
        )

        fused_output = self.fusion_norm(fused_output)


        return fused_output