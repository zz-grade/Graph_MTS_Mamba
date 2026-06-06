import torch
import torch.nn as nn
import torch.nn.functional as fun
from datetime import datetime
import math
import copy

from models.augmentation import candidate_adj_contrastive_loss_fast


class transformer_construction(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_heads):
        super(transformer_construction, self).__init__()

        self.num_heads = num_heads
        self.Query_lists = nn.ModuleList()
        self.Key_lists = nn.ModuleList()

        for i in range(num_heads):
            self.Query_lists.append(nn.Linear(input_dimension, hidden_dimension))
            self.Key_lists.append(nn.Linear(input_dimension, hidden_dimension))

    def forward(self, X):
        relation_lists = []
        for i in range(self.num_heads):
            Query_i = self.Query_lists[i](X)
            Query_i = fun.leaky_relu(Query_i)

            Key_i = self.Key_lists[i](X)
            Key_i = fun.leaky_relu(Key_i)

            rela_i = torch.bmm(Query_i, torch.transpose(Key_i, -1, -2))
            relation_lists.append(rela_i)

        relation_lists = torch.stack(relation_lists, 1)
        relation_lists = torch.mean(relation_lists, 1)
        return relation_lists



class NeuralSparseSparsifier(nn.Module):

    def __init__(self, configs, edge_num, similar_edge, node_dim, edge_dim=0, hidden=128):
        super().__init__()
        in_dim = 2 * configs.hidden_channels + edge_dim
        self.edge_num = configs.edge_num
        self.similar_edge = configs.similar_edge
        # 让 alpha 也变成可学习（可选）
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(0.1, dtype=torch.float32)))
        # 控制自学习边权影响强度
        self.log_beta = nn.Parameter(torch.log(torch.tensor(configs.log_beta, dtype=torch.float32)))

        # 根据一条边两端节点特征，学习这条边的 gate
        # 输入维度 = x_src + x_dst + |x_src-x_dst| + x_src*x_dst + dt + prior_w
        #         = 4*D + 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(configs.hidden_channels * 4 + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X, Adj, num_nodes, self_loop=False, edge_eps=0.0):
        """
        X:   (B, T*N, D)
        Adj: (B, T*N, T*N)
        num_nodes: 每个时间步的节点数 N
        """
        b_samples, total_nodes, feat_dim = X.shape
        device = X.device

        assert num_nodes is not None
        assert total_nodes % num_nodes == 0

        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()

        # -------------------------------------------------
        # 1) 先从原始 Adj 取候选边，不构造完整 time_dist / Adj_decay
        # -------------------------------------------------
        if self_loop:
            edge_mask = Adj > 0
        else:
            edge_mask = Adj > 0
            diag_mask = ~torch.eye(total_nodes, device=device, dtype=torch.bool).unsqueeze(0)
            edge_mask = edge_mask & diag_mask

        b_idx, src, dst = edge_mask.nonzero(as_tuple=True)  # (E,)

        # -------------------------------------------------
        # 2) 逐边计算时间差 dt，而不是构造 (TN, TN) 的 time_dist
        # -------------------------------------------------
        t_src = torch.div(src, num_nodes, rounding_mode='floor')
        t_dst = torch.div(dst, num_nodes, rounding_mode='floor')
        dt = (t_src - t_dst).abs().to(X.dtype)  # (E,)

        # -------------------------------------------------
        # 3) 逐边计算时间衰减后的先验边权
        # -------------------------------------------------
        adj_raw = Adj[b_idx, src, dst]  # (E,)
        # prior_w = adj_raw * torch.exp(-alpha * dt)  # (E,)
        prior_w = adj_raw * torch.exp(-alpha * dt)  # (E,)

        # 如果 edge_eps 的语义是“衰减后的边权阈值”，在这里再筛一次
        if edge_eps > 0:
            keep = prior_w > edge_eps
            b_idx = b_idx[keep]
            src = src[keep]
            dst = dst[keep]
            dt = dt[keep]
            prior_w = prior_w[keep]

        # -------------------------------------------------
        # 4) batched graph -> 大图编号
        # -------------------------------------------------
        offset = b_idx * total_nodes
        src_big = src + offset
        dst_big = dst + offset
        edge_index_big = torch.stack([src_big, dst_big], dim=0).long()

        # -------------------------------------------------
        # 5) 逐边取节点特征
        # -------------------------------------------------
        x_src = X[b_idx, src]  # (E, D)
        x_dst = X[b_idx, dst]  # (E, D)

        # -------------------------------------------------
        # 6) 自学习边权
        # -------------------------------------------------
        dt_feat = dt.unsqueeze(-1)  # (E, 1)
        prior_feat = prior_w.unsqueeze(-1)  # (E, 1)

        edge_feat = torch.cat([
            x_src,
            x_dst,
            torch.abs(x_src - x_dst),
            x_src * x_dst,
            dt_feat,
            prior_feat
        ], dim=-1)  # (E, 4D+2)

        learned_logit = self.edge_mlp(edge_feat).squeeze(-1)
        learned_gate = beta * torch.sigmoid(learned_logit)
        edge_weight = prior_w * learned_gate

        return edge_index_big, edge_weight


class TopKNeuralSparseSparsifier(nn.Module):
    """
    每个节点仅保留相似度最高的 k 条边，并学习所选边的权重。
    """

    def __init__(self, configs, hidden=128, temperature=1.0):
        super().__init__()
        self.k = configs.top_k
        self.temperature = temperature

        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(0.1, dtype=torch.float32))
        )
        self.log_beta = nn.Parameter(
            torch.log(torch.tensor(1.0, dtype=torch.float32))
        )

        # x_src, x_dst, |x_src-x_dst|, x_src*x_dst, dt, similarity
        self.edge_mlp = nn.Seuential(
            nn.Linear(4 * configs.hidden_channels + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X, Adj, num_nodes, self_loop=False):
        """
        Args:
            X: (B, M, D)，M=T*N
            Adj: (B, M, M)，节点相似度矩阵
            num_nodes: 每个时间步的节点数 N
            self_loop: 是否允许自环

        Returns:
            edge_index: (2, B*M*k)
            edge_weight: (B*M*k,)
        """
        B, M, D = X.shape
        device = X.device

        if Adj.shape != (B, M, M):
            raise ValueError(
                f"Adj shape应为 {(B, M, M)}，实际为 {tuple(Adj.shape)}"
            )

        max_k = M if self_loop else M - 1
        k = min(self.k, max_k)

        if k <= 0:
            return (
                torch.empty(2, 0, dtype=torch.long, device=device),
                torch.empty(0, dtype=X.dtype, device=device)
            )

        similarity = Adj.clone()

        if not self_loop:
            diagonal = torch.eye(
                M, dtype=torch.bool, device=device
            ).unsqueeze(0)
            similarity = similarity.masked_fill(diagonal, float("-inf"))

        # 每个节点选择最相似的 k 个节点
        topk_sim, topk_dst = torch.topk(
            similarity,
            k=k,
            dim=-1,
            largest=True,
            sorted=True
        )

        batch_idx = torch.arange(
            B, device=device
        )[:, None, None].expand(B, M, k)

        src_idx = torch.arange(
            M, device=device
        )[None, :, None].expand(B, M, k)

        x_src = X[batch_idx, src_idx]      # (B,M,k,D)
        x_dst = X[batch_idx, topk_dst]     # (B,M,k,D)

        # 时间距离
        t_src = torch.div(
            src_idx, num_nodes, rounding_mode="floor"
        )
        t_dst = torch.div(
            topk_dst, num_nodes, rounding_mode="floor"
        )
        dt = torch.abs(t_src - t_dst).to(X.dtype)

        edge_features = torch.cat(
            [
                x_src,
                x_dst,
                torch.abs(x_src - x_dst),
                x_src * x_dst,
                dt.unsqueeze(-1),
                topk_sim.unsqueeze(-1)
            ],
            dim=-1
        )

        learned_logits = self.edge_mlp(
            edge_features
        ).squeeze(-1)

        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()

        # 相似度先验 + 时间衰减 + 可学习边权
        prior_logits = (
            topk_sim / self.temperature
            - alpha * dt
        )

        edge_logits = prior_logits + beta * learned_logits

        # 每个节点的 k 条边权重和为 1
        edge_weight = fun.softmax(edge_logits, dim=-1)

        # 转成批量大图编号
        offsets = (
            torch.arange(B, device=device) * M
        )[:, None, None]

        src_big = (src_idx + offsets).reshape(-1)
        dst_big = (topk_dst + offsets).reshape(-1)

        edge_index = torch.stack(
            [src_big, dst_big],
            dim=0
        ).long()

        return edge_index, edge_weight.reshape(-1)

class NeuralSparseSparsifier_Mul(nn.Module):

    def __init__(self, configs, edge_num, similar_edge, node_dim, edge_dim=0, hidden=128):
        super().__init__()
        in_dim = 2 * configs.hidden_channels + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 共享边特征抽取
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )

        # 用于“选边”
        self.edge_score_head = nn.Linear(hidden, 1)

        # 用于“消息传递边权”
        self.edge_weight_head = nn.Linear(hidden, 1)

        self.edge_num = configs.edge_num
        self.similar_edge = configs.similar_edge
        self.random_edge = configs.random_edge
        self.max_hop = configs.max_hop
        self.ran_num = configs.ran_num
        self.sample_num = configs.sample_num

    def forward(self, X, Adj, tau=1.0, hard=False, self_loop=False, edge_eps=0.0):
        """
        返回:
          edge_index_big: (2, E)  batched edge_index，节点编号范围 [0, B*N-1]
        可选:
          edge_eps: coe_Adj > edge_eps 才认为有边（默认 0，即只要 >0 就算边）
        """
        device = X.device
        B, N, Fdim = X.shape

        # --------------------------------------------------
        # 1) 先取相似候选边
        # --------------------------------------------------
        A_sim = Adj.clone()

        K_sim = min(self.similar_edge, N)
        sim_cols = A_sim.topk(k=K_sim, dim=-1).indices  # (B, N, K_sim)

        # --------------------------------------------------
        # 2) 再采样随机候选边（不和 sim_cols/self 重复）
        # --------------------------------------------------
        K_rand = self.random_edge

        if K_rand > 0:

            # 多采一点，后面过滤重复 / self 后再截断
            sample_mul = 3
            K_pool = K_rand * sample_mul

            # 直接随机采样候选列，不再构造 (B,N,N) 的完整随机矩阵
            rand_pool = torch.randint(
                low=0,
                high=N,
                size=(B, N, K_pool),
                device=device
            )  # (B,N,K_pool)

            # allowed[b, i, j] = True 表示节点 i 可以选 j 作为随机邻居
            allowed = torch.ones(B, N, N, device=device, dtype=torch.bool)

            # 去掉 sim 邻居，避免重复
            allowed.scatter_(
                dim=-1,
                index=sim_cols,
                src=torch.zeros_like(sim_cols, dtype=torch.bool)
            )

            # 1) 去掉 self
            valid_mask = torch.ones(B, N, K_pool, device=device, dtype=torch.bool)
            if not self_loop:
                rows = torch.arange(N, device=device)[None, :, None]  # (1,N,1)
                valid_mask &= (rand_pool != rows)

            # 2) 去掉和 sim_cols 重复的随机边
            if sim_cols.size(-1) > 0:
                dup_with_sim = (rand_pool.unsqueeze(-1) == sim_cols.unsqueeze(-2)).any(dim=-1)  # (B,N,K_pool)
                valid_mask &= (~dup_with_sim)

            # 3) 先把非法位置置成 -1
            rand_pool = rand_pool.masked_fill(~valid_mask, -1)

            # 4) 去重：排序后，相邻相同元素视为重复
            rand_pool_sorted, _ = rand_pool.sort(dim=-1)  # (B,N,K_pool)

            dup_inside = torch.zeros_like(rand_pool_sorted, dtype=torch.bool)
            dup_inside[..., 1:] = (rand_pool_sorted[..., 1:] == rand_pool_sorted[..., :-1])
            rand_pool_sorted = rand_pool_sorted.masked_fill(dup_inside, -1)

            # 5) 把有效值移到前面，无效值(-1)放后面
            valid_after = rand_pool_sorted >= 0
            order = (~valid_after).long().argsort(dim=-1, stable=True)
            rand_pool_compact = rand_pool_sorted.gather(-1, order)

            # 6) 截取前 K_rand 个
            K_rand_keep = min(K_pool, K_rand * 2)
            rand_cols = rand_pool_compact[..., :K_rand_keep]

        # --------------------------------------------------
            # 3) 合并候选边：相似边 + 随机边
            # --------------------------------------------------
        cand_cols = torch.cat([sim_cols, rand_cols], dim=-1)  # (B,N,Kcand)
        Kcand = cand_cols.size(-1)

        if Kcand == 0:
            # 没候选边，直接返回空图
            edge_index_big = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_weight_big = torch.empty(0, dtype=X.dtype, device=device)
            return edge_index_big, edge_weight_big

        # cand_mask = torch.ones(B, N, Kcand, device=device, dtype=torch.bool)
        cand_mask = (cand_cols >= 0) & (cand_cols < N)
        if not self_loop:
            rows = torch.arange(N, device=device)[None, :, None].expand(B, N, Kcand)
            cand_mask = cand_mask & (cand_cols != rows)

        # 给所有 gather/index 用安全索引
        safe_cols = cand_cols.clamp(min=0, max=N - 1).long()

        # --------------------------------------------------
        # 4.5) 结构对比损失：拉开候选边内部的原始 Adj 差距
        # --------------------------------------------------
        loss_struct = candidate_adj_contrastive_loss_fast(
            Adj=Adj,
            cand_cols=safe_cols,
            cand_mask=cand_mask,
            pos_ratio=0.3,
            neg_ratio=0.3,
            margin=0.2,
            min_pos=1,
            min_neg=1
        )

        # --------------------------------------------------
        # 4) 取候选边两端节点特征，MLP 打分
        # --------------------------------------------------
        x_u = X[:, :, None, :].expand(B, N, Kcand, Fdim)  # (B,N,K,F)

        X_flat = X.reshape(B * N, Fdim)
        offset = (torch.arange(B, device=device) * N)[:, None, None]  # (B,1,1)
        flat_idx = (safe_cols + offset).reshape(-1)  # (B*N*K,)
        x_v = X_flat[flat_idx].reshape(B, N, Kcand, Fdim)  # (B,N,K,F)

        # 从 Adj 中取候选边对应的结构分数
        adj_score = Adj.gather(-1, safe_cols)   # (B,N,K)
        adj_score = adj_score.masked_fill(~cand_mask, 0.0)

        # 方法一：把边信息作为 MLP 输入，而不是后面再加
        pair = torch.cat([x_u, x_v], dim=-1)  # (B,N,K,2F+1)

        pair_feat = self.edge_mlp(pair)  # (B,N,K,H)

        # 1) 用于选边的分数
        # logits = self.edge_score_head(pair_feat).squeeze(-1)  # (B,N,K)
        sel_score = self.edge_score_head(pair_feat).squeeze(-1)

        logits = sel_score + 0.5 * adj_score

        # 2) 用于边权重的分数
        edge_weight_logits = self.edge_weight_head(pair_feat).squeeze(-1)  # (B,N,K)


        # 你可以按需要选择边权范围：
        # 方案A：sigmoid -> (0,1)
        edge_weight_pred = torch.sigmoid(edge_weight_logits)

        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~cand_mask, neg_inf)

        # 为了保险，边权也在非法位置清零
        edge_weight_pred = edge_weight_pred.masked_fill(~cand_mask, 0.0)



        # --------------------------------------------------
        # 5) Gumbel 采样 / top-k 选最终边
        # --------------------------------------------------
        g = sample_gumbel(logits.shape, device=device)
        y = (logits + g) / max(tau, 1e-8)

        k_pick = min(self.edge_num, Kcand)

        if k_pick == 0:
            edge_index_big = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_weight_big = torch.empty(0, dtype=X.dtype, device=device)
            return edge_index_big, edge_weight_big

        if hard:
            # 只根据“选边分数”选边
            pick = y.topk(k=k_pick, dim=-1).indices  # (B,N,k_pick)
            final_cols = cand_cols.gather(-1, pick)  # (B,N,k_pick)

            # 选中的边，其权重由 weight head 给出
            final_weights = edge_weight_pred.gather(-1, pick)  # (B,N,k_pick)
        else:
            # soft 情况下仍然先取 top-k 边
            pick = y.topk(k=k_pick, dim=-1).indices
            final_cols = cand_cols.gather(-1, pick)

            # 边权重仍由 MLP 输出
            final_weights = edge_weight_pred.gather(-1, pick)

            # 如果你希望 soft 模式下保留“被选中的概率”影响，
            # 可以把选择概率乘到边权上：
            #
            # select_prob = F.softmax(y, dim=-1)
            # select_prob = (select_prob * float(self.edge_num)).clamp(max=1.0)
            # final_select_prob = select_prob.gather(-1, pick)
            # final_weights = final_weights * final_select_prob

            # --------------------------------------------------
            # 6) 转成 batched edge_index
            # --------------------------------------------------
        rows = torch.arange(N, device=device)[None, :, None].expand(B, N, k_pick)

        valid = final_cols >= 0
        if not self_loop:
            valid = valid & (final_cols != rows)

        b_idx, src, kk = valid.nonzero(as_tuple=True)
        dst = final_cols[b_idx, src, kk]
        ew = final_weights[b_idx, src, kk]

        offset2 = b_idx * N
        src_big = src + offset2
        dst_big = dst + offset2

        edge_index_big = torch.stack([src_big, dst_big], dim=0).long()
        edge_weight_big = ew.float()

        # --------------------------------------------------
        # 7) 统一补 self-loop
        # --------------------------------------------------
        if self_loop:
            loop_nodes = torch.arange(B * N, device=device)
            loop_edge_index = torch.stack([loop_nodes, loop_nodes], dim=0)  # (2, B*N)

            # 自环权重
            loop_edge_weight = torch.ones(B * N, device=device, dtype=edge_weight_big.dtype)

            # 拼接到原图
            edge_index_big = torch.cat([edge_index_big, loop_edge_index], dim=1)
            edge_weight_big = torch.cat([edge_weight_big, loop_edge_weight], dim=0)

        return edge_index_big, edge_weight_big, loss_struct





class Feature_extractor_1DCNN_Tiny(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size=3, stride=1, dropout=0.2):
        super(Feature_extractor_1DCNN_Tiny, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden * 2, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(num_hidden * 2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden * 2, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # 如果通道不一致，用1x1卷积对齐
        if input_channels != out_dim:
            self.shortcut = nn.Conv1d(input_channels, out_dim, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.positional_encoding = PositionalEncoding(num_hidden, 0.1)

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)


        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = x + torch.Tensor(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x


def Dot_Graph_Construction(node_features, device):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N, device=device).repeat(bs, 1, 1)

    eyes_like_inf = eyes_like * 1e8

    Adj = fun.leaky_relu(Adj - eyes_like_inf)

    # Adj = fun.softmax(Adj, dim=-1)

    topk_val, topk_idx = torch.topk(Adj, k=10, dim=-1)  # (bs, N, K)

    mask = torch.zeros_like(Adj)
    mask.scatter_(-1, topk_idx, 1.0)

    Adj = Adj * mask

    # 可选：重新归一化（很重要！）
    Adj = Adj / (Adj.sum(dim=-1, keepdim=True) + 1e-8)

    Adj = Adj + eyes_like

    return Adj


class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, normalize=True):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )
        self.normalize = normalize

    def forward(self, node_features):
        device = node_features.device
        B, N, D = node_features.size()

        z = self.mapping(node_features) + node_features

        z = z / (D ** 0.5)

        Adj = torch.bmm(z, z.transpose(1, 2))

        eye = torch.eye(N, device=z.device, dtype=torch.bool)[None]
        Adj = Adj.masked_fill(eye, -1e4)

        # temperature + softmax
        Adj = Adj / 0.5
        Adj = fun.softmax(Adj, dim=-1)

        return Adj



def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    X_0 = X.unsqueeze(-3)
    X_1 = X.unsqueeze(-2)

    X_distance = torch.sum((X_0 - X_1)**2, -1)

    Loss_GL_0 = X_distance*Adj
    Loss_GL_0 = torch.mean(Loss_GL_0)

    Loss_GL_1 = torch.sqrt(torch.mean(Adj**2))
    # print('Loss GL 0 is {}'.format(Loss_GL_0))
    # print('Loss GL 1 is {}'.format(Loss_GL_1))


    Loss_GL = Loss_GL_0 + gamma*Loss_GL_1


    return Loss_GL


def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = torch.ones(num_node * time_length, num_node * time_length).cuda()
    for i in range(time_length):
        v = 0
        for r_i in range(i,time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1

    return Adj



def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)

    y_ = fun.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = torch.transpose(y_, 1,-1)

    return y_


def sample_gumbel(shape, device, eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U.clamp(min=eps, max=1 - eps)))


# ====== 温度 tau 的一个常用退火（从大到小）:contentReference[oaicite:9]{index=9}
def tau_anneal(epoch, r=1e-2, tau_min=0.05):
    return max(tau_min, float(torch.exp(torch.tensor(-r * epoch))))