import torch
import torch.nn as nn
import torch.nn.functional as fun
from datetime import datetime
import math
import copy


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

    def __init__(self, edge_num, similar_edge, node_dim, edge_dim=0, hidden=128):
        super().__init__()
        in_dim = 2 * node_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.edge_num = edge_num
        self.similar_edge = similar_edge


    def forward(self, X, Adj, tau=1.0, hard=True, self_loop=False):
        """
        X:   (B, N, d_n) node features
        Adj: (B, N, N)   adjacency mask (0/1) OR
             (B, N, N, d_e) edge features (0 for non-edges)
        k:   keep at most k outgoing edges per node
        tau: temperature (论文建议训练时从大到小退火) :contentReference[oaicite:5]{index=5}
        hard:
          - True: hard top-k mask (0/1)
          - False: soft weights (continuous)
        return:
          mask: (B, N, N)  (float in [0,1])
          (optional) Adj_sparse if Adj has edge features
        """
        b_samples, num_nodes, _ = X.shape
        device = X.device

        topk = Adj.topk(self.similar_edge, dim=-1).indices

        # _, idx = torch.sort(Adj, descending=True, dim=-1)
        new_coe_Adj = torch.zeros_like(Adj)
        # topk = idx[:, :, :7]
        bat_id = torch.arange(Adj.size(0)).unsqueeze(1).unsqueeze(1)
        sensor_id = torch.arange(Adj.size(1)).unsqueeze(1).unsqueeze(0)
        new_coe_Adj[bat_id, sensor_id, topk] = 1


        # Build edge mask E (B,N,N)
        edges = Adj.bool()
        edge_feat = None

        # print(datetime.now(), "节点边扩展开始")
        # Pair features (dense): (B,N,N,dn) + (B,N,N,dn) (+ edge_feat)
        Xu = X.unsqueeze(2).expand(b_samples, num_nodes, num_nodes, X.size(-1))  # u repeated along v
        Xv = X.unsqueeze(1).expand(b_samples, num_nodes, num_nodes, X.size(-1))  # v repeated along u

        pair = torch.cat([Xu, Xv], dim=-1)
        # print(datetime.now(), "节点边扩展完成")

        # print(datetime.now(), "全连接学习边开始")
        # Edge logits z_{u,v}  (论文 Eq.4) :contentReference[oaicite:6]{index=6}
        logits = self.mlp(pair).squeeze(-1)  # (B,N,N)
        # print(datetime.now(), "全连接学习边完成")

        # Mask non-neighbors to -inf so they won't be selected
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~edges, neg_inf)

        # print(datetime.now(), "Gumbel开始")
        # Gumbel trick
        g = sample_gumbel(logits.shape, device=device)
        y = (logits + g) / max(tau, 1e-8)
        # print(datetime.now(), "Gumbel完成")

        if hard:
            # Hard top-k per node (u): sample without replacement by topk on (logits+gumbel)
            # 与论文“每个节点最多选 k 条边”一致 :contentReference[oaicite:7]{index=7}
            idx = y.topk(k=min(self.edge_num, num_nodes), dim=-1).indices  # (B,N,k)
            mask = torch.zeros((b_samples, num_nodes, num_nodes), device=device, dtype=torch.float32)
            mask.scatter_(dim=-1, index=idx, value=1.0)
            # 如果某些节点度数 < k，topk 会带进 -inf 的位置；乘 E 清掉即可
            mask = mask * edges.float()
        else:
            # Soft weights: Gumbel-Softmax over neighbors (论文 Eq.6) :contentReference[oaicite:8]{index=8}
            w = fun.softmax(y, dim=-1)  # sums to 1 over all v (non-edges≈0)
            # 让“期望保留边数≈k”：用 k 倍缩放再截断到 1（工程上的常用近似）
            mask = (w * float(self.edge_num)).clamp(max=1.0) * edges.float()

        mask[bat_id, sensor_id, topk] = 0

        coe_Adj = mask + new_coe_Adj
        # print(datetime.now(), "图学习完成")
        return coe_Adj




class Feature_extractor_1DCNN_Tiny(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size=3, stride=1, dropout=0.35):
        super(Feature_extractor_1DCNN_Tiny, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden * 2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden * 2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.positional_encoding = PositionalEncoding(num_hidden, 0.1)

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)


        x_in = self.conv_block1(x_in)
        # x_in = self.conv_block2(x_in)
        x = self.conv_block3(x_in)

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


def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()

    eyes_like_inf = eyes_like * 1e8

    Adj = fun.leaky_relu(Adj - eyes_like_inf)

    Adj = fun.softmax(Adj, dim=-1)

    Adj = Adj + eyes_like

    return Adj



class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)

        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = fun.leaky_relu(Adj - eyes_like_inf)
        Adj = fun.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

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