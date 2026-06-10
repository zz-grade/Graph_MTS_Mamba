class Config(object):
    def __init__(self):
        # 本项目预处理会把 SpokenArabicDigits_eq 的 65 裁成 64
        _base(self, num_nodes=13, seq_len=64, time_len=16, window=4,
              num_classes=10, batch_size=64, epochs=30, lr=5e-4)

def _base(c, num_nodes, seq_len, time_len, window, num_classes,
          batch_size, epochs, lr, hidden=64, token=64, dropout=0.1,
          supcon=0.0, compare=0.2):
    assert time_len * window == seq_len

    c.input_channels = num_nodes
    c.num_nodes = num_nodes
    c.window_size = window
    c.time_denpen_len = time_len
    c.window_size_train = window
    c.time_denpen_len_train = time_len
    c.window_size_val = window
    c.time_denpen_len_val = time_len

    c.kernel_size = 4
    c.stride = 1
    c.convo_time_length = time_len

    c.hidden_channels = hidden
    c.final_out_channels = 64
    c.dimension_token = token
    c.dimension_state = 64
    c.dimension_conv = 4
    c.expand = 2
    c.num_local_layers = 1
    c.num_global_layers = 1

    c.num_classes = num_classes
    c.dropout = dropout
    c.num_epoch = epochs
    c.beta1 = 0.9
    c.beta2 = 0.99
    c.lr = lr
    c.weight_decay = 3e-4

    c.drop_last = False
    c.batch_size = batch_size
    c.batch_size_test = batch_size

    c.wavelet_aug = True
    c.random_aug = True
    c.Context_Cont = Context_Cont_configs()
    c.augmentation = augmentations()

    c.use_mpnn = 1
    c.mpnn_layer = 5
    c.max_hop = 6
    c.ran_num = 3
    c.repeat_sample = 1
    c.sample_num = 1
    c.show_interval = 1
    c.decay_rate = 0.7

    total_nodes = num_nodes * time_len
    c.mlp_hidden = token * 4
    c.top_k = min(8, total_nodes - 1)
    c.edge_num = min(8, total_nodes - 1)
    c.similar_edge = min(3, total_nodes - 1)
    c.random_edge = min(8, total_nodes - 1)
    c.num_anchors = min(8, num_nodes)
    c.epsilon = 20
    c.log_beta = 3.0
    c.supcon_weight = supcon
    c.compare_weight = compare
    c.similarity_threshold = 0.3
    c.graph_mode = "true"
    c.mpnn_message_scales = None


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.3
        self.jitter_ratio = 0.8
        self.max_seg = 2
        self.wavelet_aug_coeffi_weak = 0.003
        self.wavelet_aug_coeffi_strong = 0.008


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True