import torch
import torch.nn as nn
import torch.nn.functional as fun
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


class Feature_extractor_1DCNN_RUL(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size=3, stride=1, dropout=0):
        super(Feature_extractor_1DCNN_RUL, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)
        x = torch.transpose(x_in, -1, -2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x





def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()

    eyes_like_inf = eyes_like*1e8

    Adj = fun.leaky_relu(Adj-eyes_like_inf)

    Adj = fun.softmax(Adj, dim = -1)

    Adj = Adj+eyes_like

    return Adj