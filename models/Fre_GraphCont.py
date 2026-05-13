import torch
import torch.nn as nn
import torch.nn.functional as fun


class FreqGraphEncoder(nn.Module):
    def __init__(self, signal_length, pred_length, num_ts, hidden_size, approx, s, lev,
                 filter_kind="haar"):
        super().__init__()

        self.pred_length = pred_length   # 预测长度
        self.signal_len = signal_length   # FFT宽度
        self.num_ts = num_ts   # 传感器数量
        self.hidden_size = hidden_size   # 图卷积输出特征维度
        self.approx = approx   # 一些参数
        self.s = s
        self.lev = lev

        self.f_num = self.signal_len // 2 + 1
        self.num_node = self.f_num * self.num_ts

        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_length)

        num_filters = (2 * lev) if filter_kind == "haar" else (3 * lev)

    def flatten_nodes(self, feat_bfnd: torch.Tensor):
        """
        feat_bfnd: [B, F, N, D]
        return:
          Xnode: [B, M, D] where M = F*N
        """
        B, Freq, N, D = feat_bfnd.shape
        Xnode = feat_bfnd.permute(0, 2, 1, 3).contiguous()  # [B, N, F, D]
        Xnode = Xnode.view(B, N * Freq, D)  # [B, M, D]
        return Xnode


    def knn_graph_cosine(self, Xnode: torch.Tensor, k: int):
        """
        Xnode: [B, M, D]
        returns:
          edge_index: [2, E]  (src, dst) with batch flattened offsets
          edge_weight: [E]
        """
        B, M, D = Xnode.shape
        k = min(k, M - 1)
        device = Xnode.device

        Xn = fun.normalize(Xnode, dim=-1, eps=1e-12)  # [B,M,D]
        sim = torch.matmul(Xn, Xn.transpose(-1, -2))  # [B,M,M]
        sim = sim - 1e9 * torch.eye(M, device=device)[None]  # remove self

        vals, idx = torch.topk(sim, k=k, dim=-1)  # [B,M,k]

        src = torch.arange(M, device=device)[None, :, None].expand(B, M, k)
        dst = idx
        w = vals

        # flatten + batch offsets
        offsets = (torch.arange(B, device=device) * M)[:, None, None]
        src = (src + offsets).reshape(-1)
        dst = (dst + offsets).reshape(-1)
        w = w.reshape(-1)

        edge_index = torch.stack([src, dst], dim=0).long()
        edge_weight = w.float()
        return edge_index, edge_weight


    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        trend_init = trend_init.permute(0, 2, 1, 3)  # [B, N, T, D]
        seasonal_init = seasonal_init.permute(0, 2, 1, 3)  # [B, N, T, D]
        trend_output = self.Linear_Trend(trend_init)  # [B, N, T, D_out]
        x_f = torch.fft.rfft(x, n=self.signal_len, dim=1, norm='ortho')
        # x_f: [B, F, N, D] (complex), F = self.signal_len//2 + 1

        B, F, N, D = x_f.shape
        # 把 (n, f) 作为图节点：先把变量维提到前面
        x_f = x_f.permute(0, 2, 1, 3).contiguous()
        # x_f: [B, N, F, D] (complex)
        # 展平节点：一个样本里节点数 = N*F，每个节点特征维 = D（complex）
        x_nodes_c = x_f.reshape(B, N * F, D)
        # x_nodes_c: [B, N*F, D] (complex)

        batch_total_nodes = B * self.num_nodes

        # construct batch laplacians by x real part
        d_list = self.construct_laplacian(
            torch.cat((x.real, x.imag), dim=-1),
            self.k, num_nodes=batch_total_nodes
        )
        d_list = self.get_operator(d_list, self.approx, self.s, self.J,
                                   self.lev, self.device)
    # edge_index, edge_weight = self.knn_graph_cosine(Xnode, k=10)
    #     B, M, Dnode = Xnode.shape
    #     Xnode = Xnode.reshape(B * M, Dnode)
    #     Freq = feat.shape[1]
    #     N = feat.shape[2]
    #     return Xnode, edge_index, edge_weight


def create_filter(num_nodes):
    # param_clamp = ParameterClamp()
    filter = nn.Parameter(torch.Tensor(num_nodes, 1))
    nn.init.normal_(filter, mean=1, std=0.1)
    # param_clamp.apply(filter)
    return filter

def pad_to_pow2(x, dim=1):
    """
    x: [B,T,N,D]
    pad T to next power of two
    """
    T = x.shape[dim]
    T_pad = 1 << (T - 1).bit_length()  # next power of 2
    if T_pad == T:
        return x, T

    pad_len = T_pad - T
    pad_shape = [0, 0] * (x.dim() - dim - 1) + [0, pad_len]
    # pad only time dimension
    x = fun.pad(x, pad=(0, 0, 0, 0, 0, pad_len))
    return x, T