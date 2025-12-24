import torch.nn as nn
import torch
import random
from sympy import totient
from datetime import datetime
import os, threading, traceback, torch

from models.GraphMamba import GraphMambaGMN
from models.Graph_construction import Feature_extractor_1DCNN_Tiny, Dot_Graph_Construction, Conv_GraphST, Mask_Matrix, \
    NeuralSparseSparsifier
from models.augmentation import disturbance_correlations

from torch.utils.checkpoint import checkpoint
import traceback


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
        self.sparseEdge = NeuralSparseSparsifier(configs.hidden_channels, configs.edge_num, configs.similar_edge)
        self.Graph_Mamba = GraphMambaGMN(configs, args)
        self.logits = nn.Linear(configs.convo_time_length * configs.final_out_channels * configs.num_nodes,
                                configs.num_classes)
        self.seed = args.seed

    def forward(self, data, num_remain=None):
        # data = torch.transpose(data, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        # data = torch.reshape(data, [b_samples * num_nodes, time_length, dimension])
        # data = self.nonlin_map(data)
        # TS_input = torch.transpose(data, 1, 2)
        # TS_output = self.Time_Graph_contrustion(TS_input)  # size is (bs, out_dimension*num_node, tlen)
        # TS_output = torch.transpose(TS_output, 1, 2)  # size is (bs, tlen, out_dimension*num_node
        # GC_input = torch.reshape(TS_output, [b_samples, -1, num_nodes, self.hidden_dim])
        # GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        # if not hasattr(self, "_fw_cnt"):
        #     self._fw_cnt = 0
        # self._fw_cnt += 1
        #
        # print(f"FW#{self._fw_cnt} pid={os.getpid()} tid={threading.get_ident()} "
        #       f"dev={data.device} cur_dev={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'} "
        #       f"is_compiling={hasattr(torch, 'compiler') and torch.compiler.is_compiling()}")
        #
        # if self._fw_cnt <= 3:
        #     print("CALLER:\n", "".join(traceback.format_stack(limit=6)))

        b_samples, time_length, num_nodes, dimension = data.size()  # Size of data is (bs, time_length, num_nodes, dimension)
        data = torch.transpose(data, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        data = torch.reshape(data, [b_samples * num_nodes, time_length, dimension])
        # print(datetime.now(), "训练开始--------------------------------------------------")
        data = self.nonlin_map(data)
        TS_input = torch.transpose(data, 1, 2) # (b_samples, num_nodes, dimension, time_length)
        TS_output = self.Time_Graph_contrustion(TS_input)  # size is (b_samples,  num_nodes * dimension, tlen)
        TS_output = torch.transpose(TS_output, 1, 2)  # size is (b_samples, tlen, num_nodes * dimension)
        # print(datetime.now(), "节点嵌入完成")

        GC_input = torch.reshape(TS_output, [b_samples, -1, num_nodes, self.hidden_dim])  # size is (b_samples, tlen, num_nodes, dimension)
        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])  # size is (b_samples * tlen, num_nodes, dimension)
        Adj_input = Dot_Graph_Construction(GC_input)
        # print(datetime.now(), "构建时间戳图完成")


        # Adj_input = disturbance_correlations(Adj_input, num_remain)

        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        Adj_output = self.sparseEdge(GC_input, Adj_input)
        # print(datetime.now(), "构建时间戳图完成")
        rng = random.Random(self.seed)
        GC_output = self.Graph_Mamba(GC_input, Adj_output, rng)
        # GC_output = checkpoint(lambda x: self.Graph_Mamba(x, Adj_input), GC_input)
        GC_output = torch.reshape(GC_output, [b_samples, -1, num_nodes, self.output_dim])
        logits_input = torch.reshape(GC_output, [b_samples, -1])
        # print(datetime.now(), "mamba提取完成")
        logits = self.logits(logits_input)

        logits.std()
        return logits
