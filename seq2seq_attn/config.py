class Config:
    def __init__(self, dataset):
        self.EMB_SIZE = 256
        self.ENC_SIZE = 256
        self.DEC_SIZE = 256
        self.ATTN_SIZE = 256
        self.NUM_LAYER = 4
        self.SEED = 1234
        self.BATCH_SIZE = 16
        self.MAX_BUGGY_LEN = 200
        self.MAX_FIXED_LEN = 200
        self.EPOCH = 100
        self.DROPOUT = 0.5
        self.LR = 1e-4
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        if dataset == 'small':
            self.DICT_CODE = 500
            self.DICT_WORD = 500
        elif dataset == 'medium':
            self.DICT_CODE = 1500
            self.DICT_WORD = 1500
        elif dataset == '200':
            self.DICT_CODE = 1500
            self.DICT_WORD = 1500
        else:
            raise Exception