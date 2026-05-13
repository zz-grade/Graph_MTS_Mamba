import torch
import torch.nn.functional as fun


def disturbance_correlations(Adj, numremained):
    _, idx = torch.sort(Adj, descending=True, dim=-1)
    new_coe_Adj = torch.zeros_like(Adj)
    topk = idx[:, :, :numremained]
    bat_id = torch.arange(Adj.size(0)).unsqueeze(1).unsqueeze(1)
    sensor_id = torch.arange(Adj.size(1)).unsqueeze(1).unsqueeze(0)
    new_coe_Adj[bat_id, sensor_id, topk] = 1
    rand_coe_Adj = torch.normal(mean=1, std=1, size=Adj.size())
    rand_coe_Adj[bat_id, sensor_id, topk] = 0
    rand_coe_Adj = rand_coe_Adj.cuda() if torch.cuda.is_available() else rand_coe_Adj
    coe_Adj = new_coe_Adj + rand_coe_Adj
    return Adj * coe_Adj


def disturbance_correlations_edge_index(Adj, numremained):
    """
    Adj: (B, N, N)  相似度/相关性矩阵
    return: edge_index_big (2, E)
    """
    device = Adj.device
    B, N, _ = Adj.shape

    # 1) 取 top-k 索引（去自环）
    A = Adj.clone()
    A.diagonal(dim1=-2, dim2=-1).fill_(-1e9)

    _, idx = torch.sort(A, descending=True, dim=-1)
    topk = idx[:, :, :numremained]          # (B, N, K)

    # 2) 构造 (row, col)
    rows = torch.arange(N, device=device)[None, :, None].expand(B, N, numremained)
    cols = topk                              # (B, N, K)

    # 3) 去掉自环（保险）
    mask = rows != cols

    # 4) 加 batch offset → batched graph
    offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src = (rows + offset)[mask]
    dst = (cols + offset)[mask]

    # 5) 拼成 edge_index_big
    edge_index_big = torch.stack([src, dst], dim=0).long()

    return edge_index_big



def make_random_d_regular_edge_index(num_nodes: int, degree: int, device: torch.device) -> torch.Tensor:
    """
    Simple random regular-ish directed graph construction (not a perfect expander proof, but works as a cheap proxy).
    Returns edge_index: (2, E), with E = num_nodes * degree
    """
    # For each node u, sample 'degree' destinations (with replacement avoided via topk on random scores)
    scores = torch.rand(num_nodes, num_nodes, device=device)
    scores.fill_diagonal_(-1.0)  # avoid self loop
    dst = scores.topk(k=min(degree, num_nodes - 1), dim=-1).indices  # (A, degree)
    src = torch.arange(num_nodes, device=device).unsqueeze(-1).expand_as(dst)  # (A, degree)
    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, A*degree)
    return edge_index



def build_topk_neighbor_mask(edge_index, edge_weight, B, N, topk, device):
    """
    根据 edge_index + edge_weight 构建 (B, N, N) 的 top-k 邻居 mask
    """
    adj = torch.zeros(B, N, N, device=device)

    src, dst = edge_index  # (E,), (E,)

    graph_id = src // N
    src_local = src % N
    dst_local = dst % N

    for i in range(src.shape[0]):
        b = graph_id[i]
        adj[b, src_local[i], dst_local[i]] = edge_weight[i]

    # 无向图
    adj = torch.maximum(adj, adj.transpose(1, 2))

    # 每个节点选 top-k 邻居
    topk_idx = torch.topk(adj, k=topk, dim=-1).indices  # (B, N, topk)

    mask = torch.zeros_like(adj, dtype=torch.bool)  # (B, N, N)
    for b in range(B):
        for i in range(N):
            mask[b, i, topk_idx[b, i]] = True

    return mask


def node_contrast_topk_edge_weight_single_view(
    z,
    edge_index,
    edge_weight,
    topk=5,
    tau=0.2,
    include_self=False,
):
    """
    单视图节点对比学习（基于 edge_weight 的 top-k 邻居）

    参数:
        z:           (B, N, D)
        edge_index:  (2, E), batched graph 的边
        edge_weight: (E,)
        topk:        每个节点选多少个正邻居
        tau:         温度系数
        include_self: 是否把自身节点也算作正样本（单视图下通常建议 False）

    返回:
        loss: scalar
    """
    B, N, D = z.shape
    device = z.device

    # 1) normalize
    z = fun.normalize(z, dim=-1)  # (B, N, D)

    # 2) 单视图相似度矩阵
    sim = torch.matmul(z, z.transpose(1, 2)) / tau   # (B, N, N)

    # 3) 正样本 mask：top-k 邻居
    pos_mask = build_topk_neighbor_mask(edge_index, edge_weight, B, N, topk, device)

    # 可选：是否把自己也作为正样本
    if include_self:
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
        pos_mask = pos_mask | eye

    # 4) 单视图时，通常把 self 从分母里去掉，避免自己和自己竞争
    self_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    sim = sim.masked_fill(self_mask, -1e9)

    # 5) log-softmax
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)  # (B, N, N)

    # 6) multi-positive InfoNCE
    pos_mask = pos_mask.float()
    pos_count = pos_mask.sum(dim=-1)  # (B, N)

    # 防止某些节点没有正样本
    valid = pos_count > 0

    loss = -(log_prob * pos_mask).sum(dim=-1) / (pos_count + 1e-12)  # (B, N)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = loss[valid].mean()
    return loss



def noisy_topk(adj, k, noise_scale=0.1):
    noise = torch.randn_like(adj) * noise_scale
    adj_noisy = adj + noise

    topk_idx = torch.topk(adj_noisy, k=k, dim=-1).indices

    mask = torch.zeros_like(adj, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))

    return mask


def candidate_adj_contrastive_loss_fast(
    Adj,
    cand_cols,
    cand_mask,
    pos_ratio=0.3,
    neg_ratio=0.3,
    margin=0.2,
    min_pos=1,
    min_neg=1
):
    """
    在候选边集合内部，对 Adj 的边分数做结构对比损失（优化版）

    Args:
        Adj:       (B, N, N)
        cand_cols: (B, N, Kcand), 非法位置可为 -1
        cand_mask: (B, N, Kcand), True 表示合法候选边
    Returns:
        loss: scalar
    """
    B, N, _ = Adj.shape
    device = Adj.device
    dtype = Adj.dtype

    if cand_cols.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    Kcand = cand_cols.size(-1)
    if Kcand == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # 1) gather 候选边分数
    safe_cols = cand_cols.clamp_min(0)
    adj_cand = Adj.gather(-1, safe_cols)   # (B, N, Kcand)

    # 2) 先筛掉有效候选数不足的节点，只对有效节点算
    num_valid = cand_mask.sum(dim=-1)      # (B, N)
    valid_node = (num_valid >= 2)          # (B, N)

    if not valid_node.any():
        return torch.zeros((), device=device, dtype=dtype)

    # 只保留有效节点，压平后处理
    adj_cand = adj_cand[valid_node]        # (M, Kcand)
    cand_mask = cand_mask[valid_node]      # (M, Kcand)
    num_valid = num_valid[valid_node]      # (M,)

    M = adj_cand.size(0)
    if M == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # 3) 根据 Kcand 确定 topk 数量
    k_pos = max(min_pos, int(round(Kcand * pos_ratio)))
    k_neg = max(min_neg, int(round(Kcand * neg_ratio)))
    k_pos = min(k_pos, Kcand)
    k_neg = min(k_neg, Kcand)

    # 若某些节点有效边少于 k_pos / k_neg，后面会靠 mask 过滤
    neg_inf = torch.finfo(dtype).min
    pos_inf = torch.finfo(dtype).max

    adj_for_pos = adj_cand.masked_fill(~cand_mask, neg_inf)
    adj_for_neg = adj_cand.masked_fill(~cand_mask, pos_inf)

    # 4) 取正负样本
    pos_idx = torch.topk(adj_for_pos, k=k_pos, dim=-1, largest=True).indices    # (M, k_pos)
    neg_idx = torch.topk(adj_for_neg, k=k_neg, dim=-1, largest=False).indices   # (M, k_neg)

    pos_score = torch.gather(adj_for_pos, dim=-1, index=pos_idx)   # (M, k_pos)
    neg_score = torch.gather(adj_for_neg, dim=-1, index=neg_idx)   # (M, k_neg)

    pos_valid = torch.gather(cand_mask, dim=-1, index=pos_idx)     # (M, k_pos)
    neg_valid = torch.gather(cand_mask, dim=-1, index=neg_idx)     # (M, k_neg)

    # 5) pairwise hinge loss
    pair_valid = pos_valid.unsqueeze(-1) & neg_valid.unsqueeze(-2)  # (M, k_pos, k_neg)

    loss_mat = fun.relu(margin - pos_score.unsqueeze(-1) + neg_score.unsqueeze(-2))
    loss_mat = loss_mat * pair_valid.to(loss_mat.dtype)

    pair_count = pair_valid.sum(dim=(-1, -2)).clamp_min(1)          # (M,)
    loss_per_node = loss_mat.sum(dim=(-1, -2)) / pair_count         # (M,)

    return loss_per_node.mean()


def augment_with_adaptive_shift(data, kernel_size, ratio=0.3, noise_std=0.01):
    """
    根据卷积窗口大小，自适应时间平移

    data: (B, T, N, D)
    """

    x = data.clone()
    B, T, N, D = x.shape

    # ===== 1. 自适应 shift_range =====
    shift_range = max(1, int(kernel_size * ratio))
    # shift_range = 0

    # 限制不能超过安全范围
    shift_range = min(shift_range, kernel_size // 2)

    # ===== 2. 时间平移 =====
    if shift_range > 0:
        shifts = torch.randint(-shift_range, shift_range + 1, (B,), device=x.device)

        idx = torch.arange(T, device=x.device)[None, :]
        shifted_idx = (idx - shifts[:, None]) % T
        shifted_idx = shifted_idx[:, :, None, None].expand(B, T, N, D)

        x = torch.gather(x, dim=1, index=shifted_idx)

    # ===== 3. 小噪声 =====
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    return x


def contrastive_loss(z1, z2, tau=0.2):
    z1 = fun.normalize(z1, dim=-1)
    z2 = fun.normalize(z2, dim=-1)

    sim = torch.matmul(z1, z2.T) / tau

    labels = torch.arange(z1.size(0), device=z1.device)

    loss = fun.cross_entropy(sim, labels)
    return loss