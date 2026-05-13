import torch
import torch.nn as nn
import torch.nn.functional as fun

from models.augmentation import make_random_d_regular_edge_index


# -----------------------------
# 2) Build explicit sparse interaction graph H (edge_index_big, edge_weight)
#    H includes:
#      - node <-> anchor edges (routing)
#      - anchor <-> anchor expander edges
#      - global nodes <-> anchors
# -----------------------------
class SparseHBuilder(nn.Module):
    def __init__(self, num_anchors: int, anchor_degree: int = 6, num_global: int = 2):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_degree = anchor_degree
        self.num_global = num_global

    def forward(self, node2anchor_idx: torch.Tensor, node2anchor_w: torch.Tensor, N: int):
        """
        node2anchor_idx: (B,N,K)
        node2anchor_w:   (B,N,K)
        N: number of nodes

        We build a batched "big graph" with per-batch offsets.
        Node ids:   [0 .. N-1]
        Anchor ids: [N .. N+A-1]
        Global ids: [N+A .. N+A+G-1]
        Total per graph: N + A + G
        Big graph has B*(N+A+G) nodes.
        """
        B, _, K = node2anchor_idx.shape
        A = self.num_anchors
        G = self.num_global
        device = node2anchor_idx.device

        per_graph_nodes = N + A + G
        total_nodes = B * per_graph_nodes

        # ---------- node <-> anchor edges ----------
        b = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)  # (B,N,K)
        u = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)  # (B,N,K)
        a = node2anchor_idx  # (B,N,K)

        # local ids within each graph
        src_node = u
        dst_anchor = N + a

        # flatten
        b_flat = b.reshape(-1)
        src_flat = src_node.reshape(-1)
        dst_flat = dst_anchor.reshape(-1)
        w_flat = node2anchor_w.reshape(-1)

        # add batch offsets
        offset = b_flat * per_graph_nodes
        src_big = src_flat + offset
        dst_big = dst_flat + offset

        # make bidirectional for better mixing (node->anchor and anchor->node)
        edge_index_na = torch.stack([src_big, dst_big], dim=0)  # (2, E1)
        edge_index_an = torch.stack([dst_big, src_big], dim=0)  # (2, E1)
        edge_weight_na = w_flat
        edge_weight_an = w_flat

        # ---------- anchor <-> anchor expander edges ----------
        exp_local = make_random_d_regular_edge_index(device=device)  # (2, A*deg)
        exp_src_local = (N + exp_local[0])  # anchors local ids
        exp_dst_local = (N + exp_local[1])

        # replicate for each batch
        exp_src = exp_src_local.unsqueeze(0).expand(B, -1).reshape(-1)  # (B*A*deg,)
        exp_dst = exp_dst_local.unsqueeze(0).expand(B, -1).reshape(-1)
        exp_b = torch.arange(B, device=device).unsqueeze(1).expand(B, exp_local.size(1)).reshape(-1)
        exp_off = exp_b * per_graph_nodes

        exp_src_big = exp_src + exp_off
        exp_dst_big = exp_dst + exp_off

        edge_index_aa = torch.stack([exp_src_big, exp_dst_big], dim=0)
        edge_index_aa_rev = torch.stack([exp_dst_big, exp_src_big], dim=0)
        edge_weight_aa = torch.ones(edge_index_aa.size(1), device=device, dtype=w_flat.dtype)
        edge_weight_aa_rev = torch.ones(edge_index_aa_rev.size(1), device=device, dtype=w_flat.dtype)

        # ---------- global <-> anchors ----------
        if G > 0:
            g_ids_local = torch.arange(G, device=device) + (N + A)  # (G,)
            a_ids_local = torch.arange(A, device=device) + N  # (A,)

            # connect each global to all anchors (dense in anchors, but anchors are small)
            gg = g_ids_local.view(1, G, 1).expand(B, G, A)  # (B,G,A)
            aa = a_ids_local.view(1, 1, A).expand(B, G, A)  # (B,G,A)
            bb = torch.arange(B, device=device).view(B, 1, 1).expand(B, G, A)

            gg_f = gg.reshape(-1)
            aa_f = aa.reshape(-1)
            bb_f = bb.reshape(-1)
            off = bb_f * per_graph_nodes

            g_big = gg_f + off
            a_big = aa_f + off

            edge_index_ga = torch.stack([g_big, a_big], dim=0)
            edge_index_ag = torch.stack([a_big, g_big], dim=0)
            edge_weight_ga = torch.ones(edge_index_ga.size(1), device=device, dtype=w_flat.dtype)
            edge_weight_ag = torch.ones(edge_index_ag.size(1), device=device, dtype=w_flat.dtype)
        else:
            edge_index_ga = edge_index_ag = None
            edge_weight_ga = edge_weight_ag = None

        # concat all
        edge_indices = [edge_index_na, edge_index_an, edge_index_aa, edge_index_aa_rev]
        edge_weights = [edge_weight_na, edge_weight_an, edge_weight_aa, edge_weight_aa_rev]
        if G > 0:
            edge_indices += [edge_index_ga, edge_index_ag]
            edge_weights += [edge_weight_ga, edge_weight_ag]

        edge_index_big = torch.cat(edge_indices, dim=1)  # (2,E)
        edge_weight = torch.cat(edge_weights, dim=0)  # (E,)

        return edge_index_big, edge_weight, total_nodes, per_graph_nodes


class AnchorRouter(nn.Module):
    def __init__(self, dim: int, num_anchors: int, topk: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_anchors = num_anchors
        self.topk = topk
        self.temperature = temperature

        # learnable anchors (M, D)
        self.anchors = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)

        # routing logits: score(x_i, a_j) = (Wq x_i) · (Wk a_j)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)

    def forward(self, X: torch.Tensor, hard_topk: bool = True):
        """
        X: (B, N, D)
        Returns:
          node2anchor_idx: (B, N, K) anchor indices
          node2anchor_w:   (B, N, K) weights (softmax over K)
        """
        B, N, D = X.shape
        A = self.num_anchors
        device = X.device

        Q = self.Wq(X)  # (B, N, D)
        K = self.Wk(self.anchors).unsqueeze(0)  # (1, A, D)

        # logits: (B, N, A)  (O(B*N*A))
        logits = torch.einsum("bnd,bad->bna", Q, K.expand(B, -1, -1))
        logits = logits / max(self.temperature, 1e-6)

        # choose topK anchors per node
        Ksel = min(self.topk, A)
        topk = logits.topk(Ksel, dim=-1).indices  # (B, N, Ksel)
        topv = logits.gather(-1, topk)  # (B, N, Ksel)

        # weights over chosen anchors
        w = fun.softmax(topv, dim=-1)  # (B, N, Ksel)

        if hard_topk:
            # already hard-selected by topk; w stays soft over selected
            return topk, w
        else:
            # if you want "denser" behavior, you could return full softmax, but that loses sparsity
            return topk, w