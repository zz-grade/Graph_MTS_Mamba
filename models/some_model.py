import torch.nn as nn
import torch

from graph_construction.Graph_construction import Feature_extractor_1DCNN_RUL


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
        self.Time_Graph_contrustion = Feature_extractor_1DCNN_RUL(configs.window_size, 32,configs.hidden_channels,
                                                                  configs.kernel_size, configs.stride, configs.dropout)
        #self.Graph_Mamba =
        self.logits = nn.Linear(configs.convo_time_length * configs.final_out_channels * configs.num_nodes, configs.num_classes)


    def forward(self, X, self_supervised = False, num_remain = None):
        b_samples, time_length, num_node, dimension = X.size()
