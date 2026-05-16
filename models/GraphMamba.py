import random

import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch.utils.checkpoint import checkpoint
from datetime import datetime


from mamba_ssm import Mamba
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from models.augmentation import build_topk_neighbor_mask


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

        self.pool_score = torch.nn.Linear(d_model, 1, bias=True)

        self.gnn_pre = UFGConv(
                    in_channels=256,
                    out_channels=256,
                    channel_mix=True,
                )


        self.edge_index = None  # (2, E)
        self.tokens_per_node = None
        self.seed = args.seed
        self.configs = configs
        self.d_model = d_model
        self.use_mpnn = None
        self.b_sample = configs

        self.convo_time_length = configs.convo_time_length
        self.pool_nn = DiffPoolForBNLD_NoEmbed(64, configs.num_anchors)

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
                ht = layer(ht)
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




    def forward(self, x, edge_index_big, edge_weight_big, adj):
        device = x.device
        b_samples, num_node, dimension = x.size()
        L = self.convo_time_length
        num_node /= L

        node_token_feats_list = []
        perm_batch_list = []
        inv_perm_batch_list = []
        edge_index_list = []
        # print(datetime.now(), "mamba训练开始")
        # print(b_samples, "每批次样本数")
        node_token_feats, perm_batch, inv_perm_batch, edge_index_big = self.Graph_del(x, edge_index_big, edge_weight_big, b_samples, num_node, device)
        # node_token_feats = self.gnn_pre(x,edge_index_big,adj)
        # print(datetime.now(), "子图构建及编码完成")

        b_samples, num_node, _, dimension = node_token_feats.size()

        # # 你需要提供/保存 N 和 F（freq bins）
        # # N = num_nodes (传感器数)
        # # F = freq bins (rfft 后频点数 or keep_low_freq)
        # N = 144
        # F = 17
        #
        # # [B, M, D] -> [B, N, F, D]
        # x_nf = node_token_feats.view(b_samples, N, F, dimension)
        #
        # # ========== 1) “频率片内”：节点编码（沿 N 维）==========
        # # (B, N, F, D) -> (B*F, N, D)
        # x_intra = x_nf.permute(0, 2, 1, 3).contiguous().view(b_samples * F, N, dimension)
        # for layer in self.local_mamba:
        #     x_intra = layer(x_intra)  # (B*F, N, D)
        #
        # # 回到 (B, N, F, D)
        # x_intra = x_intra.view(b_samples, F, N, dimension).permute(0, 2, 1, 3).contiguous()
        #
        # # ========== 2) 排序（如果你仍然要按 perm 排节点序）==========
        # # perm_batch: [B, N] —— 仍然只对“传感器节点维 N”排序
        # # perm_expanded = perm_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F, dimension)  # (B,N,F,D)
        # # x_sorted = torch.gather(x_intra, 1, perm_expanded.to(x_intra.device))  # (B,N,F,D)
        #
        # # ========== 3) 全局：跨“频率 token” mamba（沿 F 维）==========
        # # (B,N,F,D) -> (B*N, F, D)
        # h_freq = x_intra.view(b_samples * N, F, dimension)
        # for layer in self.global_mamba:
        #     h_freq = layer(h_freq)  # (B*N, F, D)
        #
        # # 取最后一个频率 token（也可以改成 mean/max/attention）
        # node_repr_all = h_freq[:, -1, :]  # (B*N, D)
        # node_repr_global_1 = node_repr_all.view(b_samples, N, dimension)  # (B,N,D)

        # ========== 1) 时间片内：节点编码（沿 N 维）==========
        # (B, N, L, D) -> (B*L, N, D)
        x_intra = node_token_feats.permute(0, 2, 1, 3).contiguous().view(b_samples * L, -1, dimension)
        # x_intra = node_token_feats
        for layer in self.local_mamba:  # 你需要新加这个模块列表
            x_intra = layer(x_intra)  # 仍是 (B*L, N, D)

        # 回到 (B, N, L, D)
        x_intra = x_intra.view(b_samples, L, -1, dimension).permute(0, 2, 1, 3).contiguous()

        # ========== 2) 排序（如果你仍然要按 perm 排节点序）==========
        perm_expanded = perm_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, dimension)  # (B,N,L,D)
        x_sorted = torch.gather(x_intra, 1, perm_expanded.to(device))  # (B,N,L,D)

        # x_intra: (B, N, L, D)
        # 先把每个节点在时间上做个summary用于打分（比如取最后时刻或均值）
        # node_summary = x_intra.mean(dim=2)  # (B, N, D)

        # X_super, A_pool, S, aux_lossn = self.pool_nn(x_intra, adj)

        # ========== 3) 全局：跨时间 mamba（沿 L 维）==========
        # (B,N,L,D) -> (B*N, L, D)
        h_time = x_sorted.view(b_samples * num_node, L, dimension)
        for layer in self.global_mamba:
            h_time = layer(h_time)  # (B*N, L, D)
        node_repr_all = h_time[:, -1, :]  # (B*N, D)
        node_repr_global_1 = node_repr_all.view(b_samples, num_node, dimension)  # (B,N,D)

        # node_repr_global_2 = build_feature_perturbed_view(node_repr_global_1,0.05, 0.002)


        # print(datetime.now(), "对比学习开始")
        #
        # # print("对比学习之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        z1 = self.proj(node_repr_global_1)  # (B,N,Dproj)
        z2 = self.proj(node_repr_global_1)  # (B,N,Dproj)
        # loss_cl = node_contrast_topk_edge_weight_multi_view(
        #     z1=node_repr_global_1,  # (B, N, D)
        #     edge_index=edge_index_big,
        #     edge_weight=edge_weight_big,
        #     topk=5,
        #     tau=0.2,
        #     include_self=True,
        # )
        loss_cl = 0


        # # print(datetime.now(), "对比损失计算开始")
        # # print("计算损失函数之前", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        loss_con = node_info_nce_cross_graph_only(z2, z1)
        # print(datetime.now(), "对比损失计算完成")
        # print(datetime.now(), "对比学习结束")
        return node_repr_global_1, loss_cl

    def _merge_edge_index_list(self, edge_index_list_in, b_samples, num_node, device):
        # edge_index_list_in: list长度B，每个是 (2, E_b)
        # 返回: edge_index_big: (2, E_total)，节点索引已加offset

        edge_chunks = []
        for b in range(b_samples):
            ei = edge_index_list_in[b]
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"edge_index_list_in[{b}] must be [2,E], got {ei.shape}")
            ei = ei.to(device).long()

            offset = b * num_node
            edge_chunks.append(ei + offset)

        edge_index_big = torch.cat(edge_chunks, dim=1)  # (2, E_total)
        return edge_index_big


    def Graph_del(self, x, edge_index_big, edge_weight_big, b_samples, num_node, device):
        # x: (B, N, D)
        x = x.to(device)
        B, N, F = x.shape

        # edge_index_big: (2, E)  节点编号范围应在 [0, B*N-1]
        if edge_index_big.device != device:
            edge_index_big = edge_index_big.to(device)

        # 2) 合并节点特征： (B,N,D) -> (B*N, D)
        x_flat = x.reshape(b_samples * N, F)

        # 3) 一次跑完 MPNN：输出 (B*N, D)
        neigh_flat = self.mpnn(x_flat, edge_index_big, edge_weight_big)  # (B*N, D)

        # 4) 还原形状给后面用： (B*N, D) -> (B,N,1,D)
        node_token_feats = neigh_flat.view(b_samples, N, -1).unsqueeze(2)  # (B,N,1,D)

        # 5) perm / inv_perm
        base_perm = torch.arange(N, device=device, dtype=torch.long)
        perm_batch = base_perm.unsqueeze(0).expand(b_samples, N).contiguous()  # (B,N)
        inv_perm_batch = perm_batch.clone()  # (B,N)

        return node_token_feats, perm_batch, inv_perm_batch, edge_index_big


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


class UFGConv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            channel_mix=True,
            bias=False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.channel_mix = channel_mix

        self.linear = Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        if self.channel_mix:
            x = self.linear(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_attr):
        return edge_attr.view(-1, 1) * x_j






def node_info_nce_cross_graph_only(z1, z2, tau=0.2, chunk_size=128):
    """
    z1, z2: (B, N, D)
    正样本：(b,i) <-> (b,i)
    负样本：只来自 b' != b 的节点
    """
    b_samples, N, D = z1.shape


    idx = torch.randperm(N, device=z1.device)[:32]  # (K,)
    z1 = z1[:, idx, :]  # (B, K, D)
    z2 = z2[:, idx, :]

    BK = b_samples * 32

    # 1. 归一化
    z1 = fun.normalize(z1, dim=-1)
    z2 = fun.normalize(z2, dim=-1)

    # 2. 展平为 (BN, D)
    x1 = z1.reshape(b_samples * 32, D)
    x2 = z2.reshape(b_samples * 32, D)

    # 3. 相似度矩阵
    # logits = (x1 @ x2.T) / tau   # (BN, BN)

    # 4. 构造 graph_id
    # 第 k = b*N + i 个节点来自第 b 个图
    graph_id = torch.arange(b_samples, device=z1.device).repeat_interleave(32)

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
        total_sum = x_rows.new_zeros(())
        col_index = torch.arange(BK, device=z1.device)  # (BK,)

        # 预先转置可减少开销
        x_cols_T = x_cols.T.contiguous()

        for i0 in range(0, BK, chunk_size):
            i1 = min(i0 + chunk_size, BK)
            row_index = torch.arange(i0, i1, device=z1.device)  # (m,)

            logits = (x_rows[i0:i1] @ x_cols_T) / tau  # (m, BK)

            same_graph = (row_ids[i0:i1].unsqueeze(1) == col_ids.unsqueeze(0))  # (m,BK)
            diag = (row_index.unsqueeze(1) == col_index.unsqueeze(0))  # (m,BK)
            mask = same_graph & (~diag)
            logits = logits.masked_fill(mask, float("-inf"))

            target = labels[i0:i1]
            # 用 sum，最后除以 BK
            total_sum = total_sum + torch.nn.functional.cross_entropy(
                logits, target, reduction="sum"
            )

            # 及时释放大张量
            del logits, same_graph, diag, mask

        return total_sum / BK
    # 6. InfoNCE（双向）

    # loss_12 = fun.cross_entropy(logits, labels)
    # loss_21 = fun.cross_entropy(logits.T, labels)
    # loss_12 = ce_rows_chunked(x1, x2, graph_id, graph_id)
    loss_21 = ce_rows_chunked(x2, x1, graph_id, graph_id)

    return loss_21


class DiffPoolForBNLD_NoEmbed(nn.Module):
    def __init__(self, dim: int, out_clusters: int, hidden_dim: int = 128):
        super().__init__()
        self.pool = DiffPoolLayerNoEmbed(dim, out_clusters, hidden_dim)

    def forward(self, X_intra: torch.Tensor, A: torch.Tensor):
        # X_intra: (B,N,L,D)
        B, N, L, D = X_intra.shape

        # 用已有特征做 assignment（只需要一个 summary 来算 S）
        X_sum = X_intra.mean(dim=2)            # (B,N,D) 你也可以换成 X_intra[:,:, -1, :]
        _, A_pool, S, aux_loss = self.pool(X_sum, A)

        # 用同一个 S 去聚合所有时间片： (B,N,L,D) -> (B,M,L,D)
        X_bt = X_intra.permute(0, 2, 1, 3).contiguous().view(B * L, N, D)   # (B*L,N,D)
        S_bt = S.unsqueeze(1).expand(B, L, N, self.pool.M).contiguous().view(B * L, N, self.pool.M)
        X_super_bt = torch.bmm(S_bt.transpose(1, 2), X_bt)                  # (B*L,M,D)
        X_super = X_super_bt.view(B, L, self.pool.M, D).permute(0, 2, 1, 3).contiguous()  # (B,M,L,D)

        return X_super, A_pool, S, aux_loss


class DiffPoolLayerNoEmbed(nn.Module):
    """
    DiffPool (only assignment + pooling), no extra feature extractor.
    X: (B,N,D), A: (B,N,N)
    """
    def __init__(self, dim: int, out_clusters: int, hidden_dim: int = 128,
                 add_aux_loss: bool = True, link_pred_weight: float = 1.0, entropy_weight: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        self.M = out_clusters
        self.add_aux_loss = add_aux_loss
        self.link_pred_weight = link_pred_weight
        self.entropy_weight = entropy_weight

        # only learn assignment S
        self.assign_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_clusters),
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        B, N, D = X.shape
        assert A.shape == (B, N, N)

        Z = X  # <-- no extra feature extraction

        S_logits = self.assign_mlp(X)          # (B,N,M)
        S = fun.softmax(S_logits, dim=-1)        # (B,N,M)

        X_pool = torch.bmm(S.transpose(1, 2), Z)                 # (B,M,D)
        A_pool = torch.bmm(torch.bmm(S.transpose(1, 2), A), S)   # (B,M,M)

        aux_loss = None
        if self.add_aux_loss:
            A_recon = torch.bmm(S, S.transpose(1, 2))            # (B,N,N)
            link_loss = fun.mse_loss(A_recon, A)
            ent = -(S * (S.clamp_min(1e-9).log())).sum(dim=-1).mean()
            aux_loss = self.link_pred_weight * link_loss + self.entropy_weight * ent

        return X_pool, A_pool, S, aux_loss


def feature_dropout(x, drop_prob=0.2, training=True):
    """
    x: (B, N, D)
    返回: 与 x 同 shape
    """
    if (not training) or drop_prob <= 0.0:
        return x

    # 逐元素 dropout；也可以改成按通道 dropout
    mask = (torch.rand_like(x) > drop_prob).float()
    x_drop = x * mask

    # 可选：保持期望不变
    x_drop = x_drop / (1.0 - drop_prob + 1e-12)
    return x_drop


def feature_noise(x, noise_std=0.01, training=True):
    """
    x: (B, N, D)
    返回: 与 x 同 shape
    """
    if (not training) or noise_std <= 0.0:
        return x

    noise = torch.randn_like(x) * noise_std
    return x + noise


def build_feature_perturbed_view(
    h,
    drop_prob=0.2,
    noise_std=0.01,
    training=True,
    use_dropout=True,
    use_noise=True,
):
    """
    基于节点表示 h 构建第二个特征扰动视图

    h: (B, N, D)
    """
    z = h

    if use_dropout:
        z = feature_dropout(z, drop_prob=drop_prob, training=training)

    if use_noise:
        z = feature_noise(z, noise_std=noise_std, training=training)

    return z


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