class Config(object):
    def __init__(self):
        # model configs
        # self.input_channels = 28
        self.num_nodes = 144

        self.window_size = 31
        self.time_denpen_len = 2
        self.window_size_train = 5
        self.time_denpen_len_train = 10

        self.window_size_val = 2
        self.time_denpen_len_val = 31

        self.convo_time_length = 5
        # self.features_len = 18


        self.kernel_size = 4
        self.stride = 1

        self.hidden_channels = 128
        self.final_out_channels = 64
        #
        # self.hidden_channels = 96
        # self.final_out_channels = 64


        self.wavelet_aug = True
        self.random_aug = True



        self.num_classes = 2
        self.dropout = 0.1
        # self.features_len = 18

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 5e-4
        self.weight_decay = 3e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 32
        self.batch_size_test = 32

        self.Context_Cont = Context_Cont_configs()
        self.augmentation = augmentations()

        self.dimension_token = 64
        self.dimension_state = 64
        self.dimension_conv = 4
        self.expand = 2
        self.num_local_layers = 2
        self.num_global_layers = 2

        self.use_mpnn = 1

        self.max_hop = 6
        self.ran_num = 2
        self.repeat_sample = 1

        self.show_interval = 1

        self.decay_rate = 0.7

        self.mpnn_layer = 1
        self.edge_num = 10
        self.similar_edge = 7


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.3
        self.jitter_ratio = 0.8
        self.max_seg = 2
        # self.wavelet_aug_coeffi = 0.3
        self.wavelet_aug_coeffi_weak = 0.003
        self.wavelet_aug_coeffi_strong = 0.008


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True
