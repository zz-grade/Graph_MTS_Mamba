import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

import trainer.Token_Create as tc


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
            BidirectionalMamba(configs)
            for _ in range(configs.num_local_layers)
        ])
        self.global_mamba = nn.ModuleList([
            BidirectionalMamba(configs)
            for _ in range(configs.num_global_layers)
        ])
        if configs.use_mpnn:
            self.mpnn = MPNN_nk(d_model, d_model)
        else:
            self.mpnn = None
        self.proj = nn.Linear(64,128)


        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.seed = args.seed
        self.configs = configs
        self.d_model = d_model
        self.use_mpnn = None


    def _encode_subgraph_tokens_single(self, x_single, tokens_per_node):
        num_nodes, _ = x_single.size()
        device = x_single.device
        l_token = len(tokens_per_node[0])
        node_token_feats = torch.zeros(num_nodes, l_token, self.d_model, device=device)
        for v in range(num_nodes):
            tokens_v = tokens_per_node[v]
            for t, node_ids in enumerate(tokens_v):
                idx = torch.tensor(node_ids, dtype=torch.long, device=device)
                sub_x = x_single[idx] # (k, in_dim)
                pooled = sub_x.mean(dim=0)  # (in_dim)
                node_token_feats[v, t] = self.subgraph_encoder(pooled)
        return node_token_feats



    def forward(self, x, adj):
        device = x.device
        b_samples, num_nodes, _ = x.size()
        rng = random.Random(self.seed)
        node_token_feats_list = []
        perm_batch_list = []
        inv_perm_batch_list = []
        edge_index_list = []
        for b in range(b_samples):
            adj_b = adj[b]
            x_b = x[b]
            adj_list_b = tc._build_adj_list_from_matrix(adj_b)
            tokens_per_node_b = tc._sample_tokens_for_all_nodes(num_nodes,self.configs,adj_list_b,rng)

            # 子图编码
            node_token_feats_b = self._encode_subgraph_tokens_single(
                x_single=x_b,
                tokens_per_node=tokens_per_node_b,
            )  # (N, L, d_model)
            node_token_feats_list.append(node_token_feats_b)

            # 度排序
            weight_sums = []
            for u in range(num_nodes):
                neighbors = adj_list_b[u]
                if len(neighbors) == 0:
                    weight_sums.append(0)
                else:
                    s = adj[u, neighbors].sum().item()
                    weight_sums.append(s)
            weight_sums = torch.tensor(weight_sums, dtype=torch.float)
            perm_b = torch.argsort(weight_sums, descending=False)  # 度小在前
            inv_perm_b = torch.empty_like(perm_b)
            inv_perm_b[perm_b] = torch.arange(num_nodes, dtype=torch.long)
            perm_batch_list.append(perm_b)
            inv_perm_batch_list.append(inv_perm_b)

            if self.use_mpnn and self.mpnn is not None:
                edge_index_b = self._build_edge_index_from_matrix_2d(adj_b).to(device)
                edge_index_list.append(edge_index_b)
            else:
                edge_index_list.append(None)

        node_token_feats = torch.stack(node_token_feats_list, dim=0)  # (B, N, L, d_model)
        perm_batch = torch.stack(perm_batch_list, dim=0)  # (B, N)
        inv_perm_batch = torch.stack(inv_perm_batch_list, dim=0)  # (B, N)

        b_samples, num_nodes, L, dimension = node_token_feats.size()
        h = node_token_feats.view(b_samples * num_nodes, L, dimension)  # (B*N, L, D)

        for layer in self.local_mamba:
            h = layer(h)
        node_repr_all = h[:, -1, :] # (B*N, D)
        node_repr = node_repr_all.view(b_samples, num_nodes, dimension)
        perm_expanded = perm_batch.unsqueeze(-1).expand(-1, -1, dimension)  # (B,N,D)
        node_seq_sorted = torch.gather(node_repr, 1, perm_expanded.to(device))  # (B,N,D)

        h_global = node_seq_sorted
        for layer in self.global_mamba:
            h_global = layer(h_global)  # (B,N,D)

        inv_perm_expanded = inv_perm_batch.unsqueeze(-1).expand(-1, -1, dimension)
        node_repr_global = torch.gather(h_global, 1, inv_perm_expanded.to(device))  # (B,N,

        if self.use_mpnn and self.mpnn is not None:
            edge_index = self.edge_index.to(device)
            mpnn_out = self.mpnn(node_repr_global, edge_index)
            node_repr_global = node_repr_global + F.relu(mpnn_out)

        node_repr_global = self.proj(node_repr_global)

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
