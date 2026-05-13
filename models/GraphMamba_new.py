import torch
import torch.nn as nn
import torch.nn.functional as fun

from mamba_ssm import Mamba
from torch_scatter import scatter_add


class GraphMambaGMN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.dimension_token
        self.local_mamba = nn.ModuleList([
            BidirectionalMamba(configs)
            for _ in range(configs.num_local_layers)
        ])
        self.global_mamba = nn.ModuleList([
            BidirectionalMamba(configs)
            for _ in range(configs.num_global_layers)
        ])
        self.mpnn = MPNN_GCN_Block(dim=d_model, dropout=configs.dropout)


        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.configs = configs
        self.d_model = d_model
        self.use_mpnn = None

        self.convo_time_length = configs.convo_time_length


    def forward(self, x, edge_index_big, rng):
        device = x.device
        b_samples, num_node, _ = x.size()
        edge_index_list = []
        node_token_feats, perm_batch, inv_perm_batch, edge_index_big = self.Graph_del(x, edge_index_big, b_samples, num_node, device)

        # print(datetime.now(), "子图构建及编码完成")


        b_samples, num_node, L, dimension = node_token_feats.size()
        h = node_token_feats.view(b_samples * num_node, L, dimension)  # (B*N, L, D)
        for layer in self.local_mamba:
            h = layer(h)

        node_repr_all = h[:, -1, :] # (B*N, D)
        node_repr = node_repr_all.view(b_samples, num_node, dimension)
        perm_expanded = perm_batch.unsqueeze(-1).expand(-1, -1, dimension)  # (B,N,D)
        node_seq_sorted = torch.gather(node_repr, 1, perm_expanded.to(device))  # (B,N,D)
        # print(datetime.now(), "节点token序列处理完成")
        # print("全局mamba之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        h_global = node_seq_sorted
        # h_global = self.run_global_mamba_time_split(h_global, self.convo_time_length, self.global_mamba)
        for layer in self.global_mamba:
            h_global = layer(h_global)  # (B,N,D)

        # print(datetime.now(), "全局token序列处理完成")

        inv_perm_expanded = inv_perm_batch.unsqueeze(-1).expand(-1, -1, dimension)
        node_repr_global_1 = torch.gather(h_global, 1, inv_perm_expanded.to(device))  # (B,N,

        if self.mpnn is not None:
            new_repr_list = []
            for b in range(b_samples):
                h_b = node_repr_global_1[b]  # (N, D)
                edge_index_b = edge_index_list[b]
                if edge_index_b is not None and edge_index_b.numel() > 0:
                    mpnn_out_b = self.mpnn(h_b, edge_index_b)
                    new_repr_b = h_b + fun.relu(mpnn_out_b)
                else:
                    new_repr_b = h_b
                new_repr_list.append(new_repr_b)
            node_repr_global_1 = torch.stack(new_repr_list, dim=0)  # (B,N,D)

        # node_repr_global = self.proj(node_repr_global)

        return node_repr_global_1




class MPNN_SAGE_Block(nn.Module):
    def __init__(self, dim, dropout=0.0, eps=1e-6):
        super().__init__()
        self.msg = nn.Linear(dim, dim, bias=False)
        self.upd = nn.Linear(2*dim, dim, bias=False)  # concat(h, agg)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(self, X, edge_index, edge_weight):
        src, dst = edge_index
        m = self.msg(X[src]) * edge_weight.unsqueeze(-1)

        agg = scatter_add(m, dst, dim=0, dim_size=X.size(0))  # sum
        deg = scatter_add(torch.ones_like(dst, dtype=X.dtype), dst, dim=0, dim_size=X.size(0))
        agg = agg / (deg.unsqueeze(-1) + self.eps)  # mean 聚合

        h = self.upd(torch.cat([X, agg], dim=-1))
        X = self.norm1(X + self.dropout(h))
        X = self.norm2(X + self.dropout(self.ffn(X)))
        return X



class MPNN_GCN_Block(nn.Module):
    def __init__(self, dim, dropout=0.0, eps=1e-6):
        super().__init__()
        self.msg = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(self, X, edge_index, edge_weight):
        src, dst = edge_index
        V = X.size(0)

        # 计算入度/出度（按你的 edge_index 定义是 src->dst）
        out_deg = scatter_add(torch.ones_like(src, dtype=X.dtype), src, dim=0, dim_size=V)
        in_deg  = scatter_add(torch.ones_like(dst, dtype=X.dtype), dst, dim=0, dim_size=V)

        norm = (out_deg[src] + self.eps).rsqrt() * (in_deg[dst] + self.eps).rsqrt()
        m = self.msg(X[src]) * (edge_weight * norm).unsqueeze(-1)

        agg = scatter_add(m, dst, dim=0, dim_size=V)

        X = self.norm1(X + self.dropout(agg))
        X = self.norm2(X + self.dropout(self.ffn(X)))
        return X


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