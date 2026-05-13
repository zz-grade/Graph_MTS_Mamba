import random

import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import MessagePassing
from torch.utils.checkpoint import checkpoint
from datetime import datetime


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
            nn.Linear(configs.hidden_channels, configs.mlp_hidden),
            nn.RMSNorm(configs.mlp_hidden),
            nn.ReLU(),
            nn.Linear(configs.mlp_hidden, d_model),
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
            self.mpnn = MPNN_nk(configs.hidden_channels, d_model, configs.mpnn_layer)
        else:
            self.mpnn = None

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 256)  # Dproj=256 可改 128/512
        )


        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.seed = args.seed
        self.configs = configs
        self.d_model = d_model
        self.use_mpnn = None

        self.convo_time_length = configs.convo_time_length

    def run_global_mamba_time_split(self, h_in, T_len, num):
        # h_in: (B_eff, N, D) where B_eff = B*T
        B_eff, N, D = h_in.shape
        assert B_eff % T_len == 0, f"B_eff={B_eff} not divisible by T_len={T_len}"
        B = B_eff // T_len

        # reshape to (B, T, N, D)
        h_btnd = h_in.view(B, T_len, N, D)

        outs = []
        # 逐时间片跑 global mamba：每次输入 (B, N, D)
        for t in range(T_len):
            ht = h_btnd[:, t]  # (B, N, D)
            for layer in num:
                ht = checkpoint(layer, ht, use_reentrant=False)
            outs.append(ht)

        # stack back -> (B, T, N, D) -> (B_eff, N, D)
        h_out = torch.stack(outs, dim=1).reshape(B_eff, N, D)
        return h_out

    def _encode_subgraph_tokens_single(self, x_single, tokens_per_node):
        num_nodes, in_dim = x_single.size()
        device = x_single.device
        l_token = len(tokens_per_node[0])
        # node_token_feats = torch.zeros(num_nodes, l_token, self.d_model, device=device)
        T = num_nodes * l_token

        flat_node_ids = []
        flat_token_ids = []

        for v in range(num_nodes):
            tokens_v = tokens_per_node[v]
            for t, node_ids in enumerate(tokens_v):
                tok_id = v * l_token + t
                flat_node_ids.extend(node_ids)
                flat_token_ids.extend([tok_id] * len(node_ids))

        idx = torch.tensor(flat_node_ids, dtype=torch.long, device=device)  # (M,)
        tok = torch.tensor(flat_token_ids, dtype=torch.long, device=device)  # (M,)

        sub_x = x_single[idx].float()   # (M, in_dim)

        sum_feat = torch.zeros(T, in_dim, device=device, dtype=torch.float32)
        sum_feat.index_add_(0, tok, sub_x)  # 按 tok 累加

        count = torch.zeros(T, 1, device=device, dtype=torch.float32)
        count.index_add_(0, tok, torch.ones((tok.numel(), 1), device=device))

        pooled = sum_feat / count.clamp(min=1)  # (T, in_dim)

        encoded = self.subgraph_encoder(pooled)  # (T, d_model) 一次前向
        node_token_feats = encoded.view(num_nodes, l_token, self.d_model)  # (N, L, D)

                #
                # idx = torch.tensor(node_ids, dtype=torch.long, device=device)
                # sub_x = x_single[idx] # (k, in_dim)
                # print(datetime.now(), "第{}个节点编码的第{}个token平均池化开始, sub_x = {}".format(v, t, t))
                # pooled = sub_x.mean(dim=0)  # (in_dim)
                # print(datetime.now(), "第{}个节点的第{}个token平均池化完成, sub_x = {}".format(v, t, t))
                # print(datetime.now(), "第{}个节点的第{}个token全连接嵌入完成, sub_x = {}".format(v, t, t))
                # node_token_feats[v, t] = self.subgraph_encoder(pooled)
                # print(datetime.now(), "第{}个节点的第{}个token全连接嵌入完成, sub_x = {}".format(v, t, t))
        return node_token_feats



    def forward(self, x, adj_1, adj_2, rng):
        device = x.device
        b_samples, num_node, _ = x.size()
        node_token_feats_list = []
        perm_batch_list = []
        inv_perm_batch_list = []
        edge_index_list = []
        # print(datetime.now(), "mamba训练开始")
        # print(b_samples, "每批次样本数")
        node_token_feats_list, perm_batch_list, inv_perm_batch_list, edge_index_list = self.Graph_del(x, adj_1, b_samples, num_node, device)

        # print(datetime.now(), "子图构建及编码完成")



        node_token_feats = torch.stack(node_token_feats_list, dim=0)  # (B, N, L, d_model)
        perm_batch = torch.stack(perm_batch_list, dim=0)  # (B, N)
        inv_perm_batch = torch.stack(inv_perm_batch_list, dim=0)  # (B, N)

        b_samples, num_node, L, dimension = node_token_feats.size()
        h = node_token_feats.view(b_samples * num_node, L, dimension)  # (B*N, L, D)

        # print("mamba之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        # h = self.run_global_mamba_time_split(h, self.convo_time_length, self.local_mamba)
        for layer in self.local_mamba:
            h = checkpoint(layer, h, use_reentrant=False)

        node_repr_all = h[:, -1, :] # (B*N, D)
        node_repr = node_repr_all.view(b_samples, num_node, dimension)
        perm_expanded = perm_batch.unsqueeze(-1).expand(-1, -1, dimension)  # (B,N,D)
        node_seq_sorted = torch.gather(node_repr, 1, perm_expanded.to(device))  # (B,N,D)
        # print(datetime.now(), "节点token序列处理完成")
        # print("全局mamba之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        h_global = node_seq_sorted
        h_global = self.run_global_mamba_time_split(h_global, self.convo_time_length, self.global_mamba)
        # for layer in self.global_mamba:
        #     h_global = layer(h_global)  # (B,N,D)

        # print(datetime.now(), "全局token序列处理完成")

        inv_perm_expanded = inv_perm_batch.unsqueeze(-1).expand(-1, -1, dimension)
        node_repr_global_1 = torch.gather(h_global, 1, inv_perm_expanded.to(device))  # (B,N,

        if self.use_mpnn and self.mpnn is not None:
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

        node_token_feats_list, perm_batch_list, inv_perm_batch_list, edge_index_list = self.Graph_del(x, adj_2, b_samples, num_node, device)


        node_token_feats = torch.stack(node_token_feats_list, dim=0)  # (B, N, L, d_model)
        perm_batch = torch.stack(perm_batch_list, dim=0)  # (B, N)
        inv_perm_batch = torch.stack(inv_perm_batch_list, dim=0)  # (B, N)

        b_samples, num_node, L, dimension = node_token_feats.size()
        h = node_token_feats.view(b_samples * num_node, L, dimension)  # (B*N, L, D)
        # print("第二次mamba之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        # h = self.run_global_mamba_time_split(h, self.convo_time_length, self.local_mamba)
        for layer in self.local_mamba:
            h = checkpoint(layer, h, use_reentrant=False)
        node_repr_all = h[:, -1, :]  # (B*N, D)
        node_repr = node_repr_all.view(b_samples, num_node, dimension)
        perm_expanded = perm_batch.unsqueeze(-1).expand(-1, -1, dimension)  # (B,N,D)
        node_seq_sorted = torch.gather(node_repr, 1, perm_expanded.to(device))  # (B,N,D)
        # print(datetime.now(), "节点token序列处理完成")
        # print("第二次全局mamba之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        h_global = node_seq_sorted
        h_global = self.run_global_mamba_time_split(h_global, self.convo_time_length, self.global_mamba)
        # for layer in self.global_mamba:
        #     h_global = layer(h_global)  # (B,N,D)
        # print(datetime.now(), "全局token序列处理完成")

        inv_perm_expanded = inv_perm_batch.unsqueeze(-1).expand(-1, -1, dimension)
        node_repr_global_2 = torch.gather(h_global, 1, inv_perm_expanded.to(device))  # (B,N,

        if self.use_mpnn and self.mpnn is not None:
            new_repr_list = []
            for b in range(b_samples):
                h_b = node_repr_global_2[b]  # (N, D)
                edge_index_b = edge_index_list[b]
                if edge_index_b is not None and edge_index_b.numel() > 0:
                    mpnn_out_b = self.mpnn(h_b, edge_index_b)
                    new_repr_b = h_b + fun.relu(mpnn_out_b)
                else:
                    new_repr_b = h_b
                new_repr_list.append(new_repr_b)
            node_repr_global_2 = torch.stack(new_repr_list, dim=0)  # (B,N,D)
        # print("对比学习之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        z1 = self.proj(node_repr_global_1)  # (B,N,Dproj)
        z2 = self.proj(node_repr_global_2)  # (B,N,Dproj)
        # print("计算损失函数之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        loss_con = node_info_nce_cross_graph_only(z1, z2)
        return node_repr_global_1, loss_con


    def Graph_del(self, x, adj, b_samples, num_node, device):
        node_token_feats_list = []
        perm_batch_list = []
        inv_perm_batch_list = []
        edge_index_list = []
        for b in range(b_samples):
            adj_b = adj[b]
            x_b = x[b]
            # print(datetime.now(), "子图构建开始")
            # adj_list_b = tc._build_adj_list_from_matrix(adj_b)
            # tokens_per_node_b = tc._sample_tokens_for_all_nodes(num_nodes,self.configs,adj_list_b,rng)

            # print(datetime.now(), "子图构建完成")
            # print(datetime.now(), "子图编码开始")
            # print(datetime.now(), "GNN开始")
            neighbor_edge = tc._build_edge_index_from_matrix(adj_b)
            neigh_nodes = self.mpnn(x_b, neighbor_edge)
            # print(datetime.now(), "GNN完成")

            # 子图编码
            # node_token_feats_b = self._encode_subgraph_tokens_single(
            #     x_single=x_b,
            #     tokens_per_node=tokens_per_node_b,
            # )  # (N, L, d_model)
            node_token_feats_b = neigh_nodes.unsqueeze(1)  # (N, 1, D)
            # node_token_feats_b = torch.cat([neigh_tok, node_token_feats_b], dim=1)  # (N, L+1, D)
            node_token_feats_list.append(node_token_feats_b)
            # print(datetime.now(), "子图编码开始")

            # with torch.no_grad():
            #     # 度排序
            #     weight_sums = []
            #     for u in range(num_nodes):
            #         neighbors = adj_list_b[u]
            #         if len(neighbors) == 0:
            #             weight_sums.append(0)
            #         else:
            #             s = adj[u, neighbors].sum().item()
            #             weight_sums.append(s)
            #     weight_sums = torch.tensor(weight_sums, dtype=torch.float)
            #     perm_b = torch.argsort(weight_sums, descending=False)  # 度小在前
            #     inv_perm_b = torch.empty_like(perm_b)
            #     inv_perm_b[perm_b] = torch.arange(num_nodes, dtype=torch.long)

            # 不做度排序
            perm_b = torch.arange(num_node, device=device, dtype=torch.long)  # (N,)
            inv_perm_b = perm_b.clone()  # (N,)

            perm_batch_list.append(perm_b)
            inv_perm_batch_list.append(inv_perm_b)

            if self.use_mpnn and self.mpnn is not None:
                edge_index_b = self._build_edge_index_from_matrix_2d(adj_b).to(device)
                edge_index_list.append(edge_index_b)
            else:
                edge_index_list.append(None)
        return node_token_feats_list, perm_batch_list, inv_perm_batch_list, edge_index_list


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

    def forward(self, x, edge_index):
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            m = self.propagate(edge_index, x=x, msg_mlp=msg_mlp)
            out = upd_mlp(torch.cat([x, m], dim=-1))
            x = out
        return out

    def message(self, x_j, msg_mlp):
        # h_j: (E_total, hidden)
        return msg_mlp(x_j)

def node_info_nce_cross_graph_only(z1, z2, tau=0.2, chunk_size=128):
    """
    z1, z2: (B, N, D)
    正样本：(b,i) <-> (b,i)
    负样本：只来自 b' != b 的节点
    """
    b_samples, N, D = z1.shape

    BN = b_samples * N

    # 1. 归一化
    z1 = fun.normalize(z1, dim=-1)
    z2 = fun.normalize(z2, dim=-1)

    # 2. 展平为 (BN, D)
    x1 = z1.reshape(b_samples * N, D)
    x2 = z2.reshape(b_samples * N, D)

    # 3. 相似度矩阵
    # logits = (x1 @ x2.T) / tau   # (BN, BN)

    # 4. 构造 graph_id
    # 第 k = b*N + i 个节点来自第 b 个图
    graph_id = torch.arange(b_samples, device=z1.device).repeat_interleave(N)

    # 5. 屏蔽“同图但非自身”的位置
    # same_graph = graph_id[:, None] == graph_id[None, :]  # (BN,BN)
    # diag = torch.eye(b_samples * N, device=z1.device, dtype=torch.bool)
    # mask = same_graph & (~diag)

    # logits = logits.masked_fill(mask, float('-inf'))

    labels = torch.arange(b_samples * N, device=z1.device)
    def ce_rows_chunked(x_rows, x_cols, row_ids, col_ids):
        """
        x_rows: (BN, D) 作为 rows
        x_cols: (BN, D) 作为 cols
        row_ids/col_ids: (BN,) graph id
        返回：mean CE（等价于 F.cross_entropy 全量计算）
        """
        BN, D = x_rows.shape
        device = x_rows.device

        col_index = torch.arange(BN, device=device)  # (BN,)

        # checkpoint 要求：被 checkpoint 的函数输入必须是 Tensor
        def ce_block(x_rows_blk, row_ids_blk, x_cols_all, col_ids_all, row_index_blk, col_index_all):
            # logits: (m, BN)
            logits = (x_rows_blk @ x_cols_all.T) / tau

            # mask 同图非自身 => -inf
            same_graph = (row_ids_blk.unsqueeze(1) == col_ids_all.unsqueeze(0))  # (m, BN)
            diag = (row_index_blk.unsqueeze(1) == col_index_all.unsqueeze(0))  # (m, BN)
            mask = same_graph & (~diag)
            logits = logits.masked_fill(mask, float("-inf"))

            # target 就是全局 index
            target = row_index_blk
            return fun.cross_entropy(logits, target, reduction="sum")

        total_sum = x_rows.new_zeros(())
        for i0 in range(0, BN, chunk_size):
            i1 = min(i0 + chunk_size, BN)

            row_index = torch.arange(i0, i1, device=device)  # (m,)
            # 注意：把需要的 slice 作为 Tensor 传进 checkpoint
            loss_sum_blk = checkpoint(
                ce_block,
                x_rows[i0:i1],
                row_ids[i0:i1],
                x_cols,
                col_ids,
                row_index,
                col_index,
                use_reentrant=False,  # 新版 PyTorch 推荐
            )
            total_sum = total_sum + loss_sum_blk

        return total_sum / BN
    # 6. InfoNCE（双向）

    # loss_12 = fun.cross_entropy(logits, labels)
    # loss_21 = fun.cross_entropy(logits.T, labels)
    loss_12 = ce_rows_chunked(x1, x2, graph_id, graph_id)
    loss_21 = ce_rows_chunked(x2, x1, graph_id, graph_id)

    return 0.5 * (loss_12 + loss_21)