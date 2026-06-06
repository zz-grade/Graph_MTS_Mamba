class Config(object):
    def __init__(self):
        # UEA Libras: 2 dimensions, length 45, 15 classes.
        self.input_channels = 2
        self.num_nodes = 2

        # The loader reshapes each dimension from length 45 to (9, 5).
        self.time_denpen_len = 9
        self.window_size = 5
        self.time_denpen_len_train = 9
        self.window_size_train = 5
        self.time_denpen_len_val = 9
        self.window_size_val = 5

        # With kernel_size=3, the current CNN maps length 9 to length 10.
        self.convo_time_length = 10
        self.kernel_size = 3
        self.stride = 1

        self.hidden_channels = 64
        self.final_out_channels = 64
        self.mlp_hidden = 256

        self.wavelet_aug = True
        self.random_aug = True

        self.num_classes = 15
        self.dropout = 0.1

        # Libras has only 180 training samples, so use a moderate batch size.
        self.num_epoch = 60
        self.batch_size = 32
        self.batch_size_test = 32
        self.drop_last = False

        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight_decay = 3e-4

        self.dimension_token = 64
        self.dimension_state = 64
        self.dimension_conv = 4
        self.expand = 2
        self.num_local_layers = 1
        self.num_global_layers = 1


        self.mpnn_layer = 5
        self.epsilon = 20
        self.log_beta = 3.0
        self.supcon_weight = 0.3
        self.graph_mode = "true"
        self.edge_num = 8
        self.mpnn_message_scales = None
        self.similar_edge = 3
        self.random_edge = 8
