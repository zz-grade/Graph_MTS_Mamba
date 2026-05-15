import torch.nn as nn
import torch
from models.GraphMamba import GraphMambaGMN
from models.Graph_construction import Feature_extractor_1DCNN_Tiny, Dot_Graph_Construction, Conv_GraphST, Mask_Matrix, \
    NeuralSparseSparsifier, Dot_Graph_Construction_weights
from models.augmentation import contrastive_loss, disturbance_correlations_edge_index, \
    augment_with_adaptive_shift


class Base_model(nn.Module):
    def __init__(self, configs, args, device):
        super().__init__()
        in_dim_fea = configs.window_size
        hidden_fea = configs.hidden_channels
        out_dim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(configs.window_size, configs.window_size)
        self.window_size = configs.window_size
        self.hidden_dim = configs.hidden_channels
        self.output_dim = configs.final_out_channels
        self.time_length = configs.convo_time_length
        self.Time_Graph_contrustion = Feature_extractor_1DCNN_Tiny(configs.window_size, 32, configs.hidden_channels,
                                                                   configs.kernel_size, configs.stride, configs.dropout)
        self.sparseEdge = NeuralSparseSparsifier(configs, configs.edge_num, configs.similar_edge, configs.hidden_channels)
        self.Graph_Mamba = GraphMambaGMN(configs, args)

        self.graph_weight = Dot_Graph_Construction_weights(configs.hidden_channels)
        # self.Fre_graph = FreqGraphEncoder(10, configs.num_nodes, configs.hidden_channels, args, 10, 10)
        self.logits = nn.Linear(configs.dimension_token, configs.num_classes)
        self.seed = args.seed
        self.device = device

    def forward(self, data, num_remain=None):

        b_samples, time_length, num_nodes, dimension = data.size()  # Size of data is (bs, time_length, num_nodes, dimension)
        data = torch.transpose(data, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        data = torch.reshape(data, [b_samples * num_nodes, time_length, dimension])
        # print(datetime.now(), "训练开始--------------------------------------------------")
        data = self.nonlin_map(data)
        TS_input = torch.transpose(data, 1, 2) # (b_samples, num_nodes, dimension, time_length)
        TS_output = self.Time_Graph_contrustion(TS_input)  # size is (b_samples,  num_nodes * dimension, tlen)
        TS_output = torch.transpose(TS_output, 1, 2)  # size is (b_samples, tlen, num_nodes * dimension)
        z1 = TS_output.reshape(-1, TS_output.size(-1))

        # print(datetime.now(), "节点嵌入完成")
        # data_noise = torch.transpose(data_noise, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        # data_noise = torch.reshape(data_noise, [b_samples * num_nodes, time_length, dimension])
        # # print(datetime.now(), "训练开始--------------------------------------------------")
        # data_noise = self.nonlin_map(data_noise)
        # TS_input_noise = torch.transpose(data_noise, 1, 2)  # (b_samples, num_nodes, dimension, time_length)
        # TS_output_noise = self.Time_Graph_contrustion(TS_input_noise)  # size is (b_samples,  num_nodes * dimension, tlen)
        # TS_output_noise = torch.transpose(TS_output_noise, 1, 2)  # size is (b_samples, tlen, num_nodes * dimension)
        # z2 = TS_output_noise.reshape(-1, TS_output_noise.size(-1))

        # loss_cl = contrastive_loss(z1,z2)
        loss_cl = data.new_zeros(())
        GC_input = torch.reshape(TS_output, [b_samples, -1, num_nodes, self.hidden_dim])  # size is (b_samples, tlen, num_nodes, dimension)
        # Gc_fre, Edge_n, Edge_w = self.Fre_graph(GC_input)


        # GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])  # size is (b_samples * tlen, num_nodes, dimension)
        GC_input = torch.reshape(GC_input, [b_samples, -1, self.hidden_dim])  # size is (b_samples * tlen, num_nodes, dimension)
        # print(datetime.now(), "构建时间戳图开始")
        Adj_input = Dot_Graph_Construction(GC_input, self.device)
        # Adj_input = self.graph_weight(GC_input)
        # loss_cl = neighbor_contrastive_loss_ratio(GC_input, Adj_input)


        # GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        GC_input = torch.reshape(GC_input, [b_samples, -1, self.hidden_dim])
        Adj_output, Adj_weight = self.sparseEdge(GC_input, Adj_input, num_nodes)
        # Adj_input = disturbance_correlations_edge_index(Adj_input, 10)
        # print(datetime.now(), "构建时间戳图完成")
        # rng = random.Random(self.seed)
        # print(datetime.now(), "mamba提取开始")
        # GC_output = self.Graph_Mamba(Gc_fre, Edge_n, Edge_w)
        # for i in range(0, 6):
        #     GC_output, _ = self.Graph_Mamba(GC_output, Adj_output, Adj_weight, Adj_input)
        GC_output, graph_loss = self.Graph_Mamba(GC_input, Adj_output, Adj_weight, Adj_input)
        logits_input = GC_output.mean(dim=1)
        # print(datetime.now(), "mamba提取完成")
        logits = self.logits(logits_input)

        logits.std()
        if torch.is_tensor(graph_loss):
            loss_cl = loss_cl + graph_loss
        return logits, loss_cl


class Base_model03(nn.Module):
    def __init__(self, configs, args):
        super().__init__()
        in_dim_fea = configs.window_size
        hidden_fea = configs.hidden_channels
        out_dim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(configs.window_size, configs.window_size)
        self.window_size = configs.window_size
        self.hidden_dim = configs.hidden_channels
        self.output_dim = configs.final_out_channels
        self.time_length = configs.convo_time_length
        self.Time_Graph_contrustion = Feature_extractor_1DCNN_Tiny(configs.window_size, 32, configs.hidden_channels,
                                                                   configs.kernel_size, configs.stride, configs.dropout)
        self.sparseEdge = NeuralSparseSparsifier_Mul(configs, configs.edge_num, configs.similar_edge, configs.hidden_channels)
        self.Graph_Mamba = GraphMambaGMN(configs, args)
        self.Fre_graph = FreqGraphEncoder(10, configs.num_nodes, configs.hidden_channels, args, 10, 10)
        self.logits = nn.Linear(36864,
                                configs.num_classes)
        self.seed = args.seed

    def forward(self, data, num_remain=None):

        b_samples, time_length, num_nodes, dimension = data.size()  # Size of data is (bs, time_length, num_nodes, dimension)
        data_noise = augment_with_adaptive_shift(data, self.window_size)
        data = torch.transpose(data, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        data = torch.reshape(data, [b_samples * num_nodes, time_length, dimension])
        # print(datetime.now(), "训练开始--------------------------------------------------")
        data = self.nonlin_map(data)
        TS_input = torch.transpose(data, 1, 2) # (b_samples, num_nodes, dimension, time_length)

        data_noise = torch.transpose(data_noise, 2, 1)  # (b_samples, num_nodes, time_length, dimension)
        data_noise = torch.reshape(data_noise, [b_samples * num_nodes, time_length, dimension])
        # print(datetime.now(), "训练开始--------------------------------------------------")
        data_noise = self.nonlin_map(data_noise)
        TS_input_noise = torch.transpose(data_noise, 1, 2)  # (b_samples, num_nodes, dimension, time_length)

        TS_output = self.Time_Graph_contrustion(TS_input)  # size is (b_samples,  num_nodes * dimension, tlen)
        TS_output_noise = self.Time_Graph_contrustion(TS_input_noise)  # size is (b_samples,  num_nodes * dimension, tlen)
        TS_output = torch.transpose(TS_output, 1, 2)  # size is (b_samples, tlen, num_nodes * dimension)
        # print(datetime.now(), "节点嵌入完成")

        GC_input = torch.reshape(TS_output, [b_samples, -1, num_nodes, self.hidden_dim])  # size is (b_samples, tlen, num_nodes, dimension)
        Gc_fre, Edge_n, Edge_w = self.Fre_graph(GC_input)

        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])  # size is (b_samples * tlen, num_nodes, dimension)
        # print(datetime.now(), "构建时间戳图开始")
        Adj_input = Dot_Graph_Construction(GC_input)




        GC_input = torch.reshape(GC_input, [-1, num_nodes, self.hidden_dim])
        Adj_output = self.sparseEdge(GC_input, Adj_input)
        # Adj_input = disturbance_correlations_edge_index(Adj_input, 10)
        # # print(datetime.now(), "构建时间戳图完成")
        # rng = random.Random(self.seed)
        # print(datetime.now(), "mamba提取开始")
        GC_output = self.Graph_Mamba(Gc_fre, Edge_n, Edge_w)
        # GC_output = checkpoint(lambda x: self.Graph_Mamba(x, Adj_input), GC_input)
        GC_output = torch.reshape(GC_output, [b_samples, -1, num_nodes, self.output_dim])
        logits_input = torch.reshape(GC_output, [b_samples, -1])
        # print(datetime.now(), "mamba提取完成")
        logits = self.logits(logits_input)

        logits.std()
        return logits


class Base_model02(nn.Module):
    def __init__(self, configs, args):
        super().__init__()

        num_classes = configs.num_classes


        self.time_length = configs.max_time_length
        self.anchorNum = configs.anchor_t_k
        self.d_model = configs.anchor_dim

        self.time_embed = Feature_extractor_1DCNN_Tiny(configs.window_size, 32, 64)
        self.getAnchor = RouteTime(configs)
        self.augAnchor = CBiasEdge(configs.d_hidden)

        # Intra-slice encoder
        self.intra = IntraSliceEncoder(configs)

        self.use_time_pos_emb = configs.use_time_pos_emb
        if self.use_time_pos_emb:
            self.time_pos_emb = nn.Embedding(self.time_length, self.d_model)
        else:
            self.time_pos_emb = None

        # Time-axis sequence model (stack of blocks)
        self.time_blocks = nn.ModuleList([
            Bidirectional_new_Mamba(configs.mamba_token, configs.mamba_state, configs.mamba_expand)
            for _ in range(configs.mamba_layer)
        ])
        self.time_norm = nn.LayerNorm(self.d_model)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.t_class_drop),
            nn.Linear(self.d_model, num_classes),
        )

        self.nonlin_map = nn.Linear(configs.window_size, configs.window_size)



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
        data = self.nonlin_map(data)
        data = torch.transpose(data, 1, 2)  # (b_samples, num_nodes, dimension, time_length)
        data = self.time_embed(data)    # Size of data is (bs, time_length, num_nodes, dimension)
        # data = data.permute(0, 2, 1, 3).contiguous()        # Size of data is (bs, num_nodes, time_length, dimension)

        data = torch.reshape(data, [b_samples, num_nodes, 128, -1])  # size is (b_samples,  num_nodes, dimension, tlen)
        data = torch.transpose(data, 2, 3)  # size is (b_samples, tlen, num_nodes * dimension)
        anchors, node_anchor, node_anchor_w = self.getAnchor(data)
        z_anchors = aggregate_nodes_to_anchors_topk(data, node_anchor, node_anchor_w, self.anchorNum)
        # idx_c = select_change_points(data)
        # node_anchor_w_new = self.augAnchor(data, z_anchors, node_anchor, node_anchor_w, idx_c)
        # z_anchors_aug = aggregate_nodes_to_anchors_topk(data, node_anchor, node_anchor_w_new, self.anchorNum)

        z_graph = self.intra(z_anchors)
        if self.use_time_pos_emb:
            t_ids = torch.arange(time_length, device=z_graph.device).view(1, time_length)
            z_graph = z_graph + self.time_pos_emb(t_ids)

        # time-axis blocks
        for blk in self.time_blocks:
            z_graph = blk(z_graph)
        z_graph = self.time_norm(z_graph)
        z = z_graph.mean(dim=1)
        # sequence -> single vector
        # if mask is None:
        #     # use mean pooling over time
        #     z = z_graph.mean(dim=1)
        # else:
        #     # masked mean
        #     mask_f = mask.float().unsqueeze(-1)  # (B,T,1)
        #     denom = mask_f.sum(dim=1).clamp_min(1.0)
        #     z = (h * mask_f).sum(dim=1) / denom

        logits = self.head(z)
        return logits

