

class Config:
    def __init__(self):
        self.seed = 1234
        self.data_dir = 'data'
        self.decom_method = 'prophet'
        if self.decom_method == 'prophet+':
            self.num_features = 17
        else:
            self.num_features = 3
        self.mdoel_saving_dir = 'ckpt'

        # shared params
        self.max_epochs = 1000
        self.min_epochs = 5
        self.batch_size = 64

        # normal
        self.normal_hidden_size = 16
        self.normal_num_layers = 1
        self.normal_lr = 1e-3
        # GRU-atten
        self.atten_hidden_size = 16
        self.atten_num_layer = 2
        self.atten_lr = 1e-3




config = Config()