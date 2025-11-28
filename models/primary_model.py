import torch.nn as nn
import torch
from sympy import totient

from models.GraphMamba import GraphMambaGMN
from models.Graph_construction import Feature_extractor_1DCNN_Tiny, Dot_Graph_Construction
from models.augmentation import disturbance_correlations


class Base_model(nn.Module):
    def __init__(self, configs, args):
        super().__init__()
        in_dim_fea = configs.window_size
        hidden_fea = configs.hidden_channels
        out_dim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(configs.window_size, configs.window_size)
        self.hidden_dim = configs.hidden_channels
        self.output_dim = configs.final_out_channels
        self.time_length = configs.convo_time_length
        self.Time_Graph_contrustion = Feature_extractor_1DCNN_Tiny(configs.window_size, 32, configs.hidden_channels,
                                                                  configs.kernel_size, configs.stride, configs.dropout)
        self.Graph_Mamba = GraphMambaGMN(configs, args)
        self.logits = nn.Linear(configs.convo_time_length * configs.final_out_channels * configs.num_nodes,
                                configs.num_classes)

    def forward(self, data, num_remain=None):
        b_samples, time_length, num_nodes, dimension = data.size()  # Size of data is (bs, time_length, num_nodes, dimension)
        data = torch.transpose(data, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        data = torch.reshape(data, [b_samples * num_nodes, time_length, dimension])
        data = self.nonlin_map(data)
        TS_input = torch.transpose(data, 1, 2)
        TS_output = self.Time_Graph_contrustion(TS_input)  # size is (bs, out_dimension*num_node, tlen)
        TS_output = torch.transpose(TS_output, 1, 2)  # size is (bs, tlen, out_dimension*num_node

        GC_input = torch.reshape(TS_output, [b_samples, -1, num_nodes, self.hidden_dim])
        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        Adj_input = Dot_Graph_Construction(GC_input)

        Adj_input = disturbance_correlations(Adj_input, num_remain)
        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        GC_output = self.Graph_Mamba(GC_input, Adj_input)
        GC_output = torch.reshape(GC_output, [b_samples, -1, num_nodes, self.output_dim])
        logits_input = torch.reshape(GC_output, [b_samples, -1])
        logits = self.logits(logits_input)

        return logits
