import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import MessagePassing




class GraphGNN(nn.Module):
    def __init__(self, configs, args):
        super().__init__()
        d_model = configs.dimension_token

        if configs.use_mpnn:
            self.mpnn = MPNN_nk(configs.hidden_channels, d_model, configs.mpnn_layer)
        else:
            self.mpnn = None



        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.seed = args.seed
        self.configs = configs
        self.d_model = d_model
        self.use_mpnn = None



    def forward(self, x, edge_index_big, edge_weight_big):
        # x: (B, N, F)
        device = x.device
        b_samples, num_node, F = x.shape

        # edge_index_big: (2, E)  节点编号范围应在 [0, B*N-1]
        if edge_index_big.device != device:
            edge_index_big = edge_index_big.to(device)

        # 2) 合并节点特征： (B,N,F) -> (B*N, F)
        x_flat = x.reshape(b_samples * num_node, F)

        # 3) 一次跑完 MPNN：输出 (B*N, D)
        neigh_flat = self.mpnn(x_flat, edge_index_big, edge_weight_big)  # (B*N, D)

        # 4) 还原形状给后面用： (B*N, D) -> (B,N,1,D)
        node_token_feats = neigh_flat.view(b_samples, num_node, -1).unsqueeze(2)  # (B,N,1,D)

        # 5) perm / inv_perm
        base_perm = torch.arange(num_node, device=device, dtype=torch.long)
        perm_batch = base_perm.unsqueeze(0).expand(b_samples, num_node).contiguous()  # (B,N)
        inv_perm_batch = perm_batch.clone()  # (B,N)

        return node_token_feats, perm_batch, inv_perm_batch, edge_index_big




class MPNN_nk(MessagePassing):
    def __init__(self, in_fea, out_fea, mpnn_layer):
        super().__init__()
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_fea, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, out_fea)
            )
            for _ in range(mpnn_layer)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
            nn.Linear(in_fea+out_fea, out_fea),
            nn.ReLU(),
            nn.Linear(out_fea, out_fea),
            )
            for _ in range(mpnn_layer)
        ])

        # 残差投影（防止维度不一致）
        if in_fea != out_fea:
            self.res_proj = nn.Linear(in_fea, out_fea)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            identity = self.res_proj(x)
            m = self.propagate(edge_index, x=x, edge_weight=edge_weight, msg_mlp=msg_mlp)
            out = upd_mlp(torch.cat([x, m], dim=-1))
            x = out + identity
        return x

    def message(self, x_j, edge_weight, msg_mlp):
        return msg_mlp(x_j) * edge_weight.unsqueeze(-1)