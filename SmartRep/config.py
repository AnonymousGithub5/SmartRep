class Config:
    def __init__(self, dataset):
        self.EMB_SIZE = 512
        self.ENC_SIZE = 512
        self.DEC_SIZE = 512
        self.ATTN_SIZE = 512

        if dataset == 'small' or dataset == 'medium':
            self.NUM_LAYER = 3
        elif dataset == 'total':
            self.NUM_LAYER = 4
        self.SEED = 2022
        self.BATCH_SIZE = 16
        self.MAX_BUGGY_LEN = 400
        self.MAX_FIXED_LEN = 600
        self.EPOCH = 250
        self.DROPOUT = 0.5
        self.LR = 1e-4
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        if dataset == 'small':
            self.DICT_CODE = 500
            self.DICT_WORD = 500
        elif dataset == 'total' or dataset == 'medium':
            self.DICT_CODE = 1500
            self.DICT_WORD = 1500

class Config_src3line:
    def __init__(self, dataset):
        self.EMB_SIZE = 256
        self.ENC_SIZE = 256
        self.DEC_SIZE = 256
        self.ATTN_SIZE = 256
        self.NUM_LAYER = 3
        self.SEED = 2022
        self.BATCH_SIZE = 16
        self.MAX_BUGGY_LEN = 400
        self.MAX_FIXED_LEN = 600
        self.EPOCH = 150
        self.DROPOUT = 0.5
        self.LR = 1e-5
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        if dataset == 'small':
            self.DICT_CODE1 = 600
            self.DICT_WORD1 = 600
            self.DICT_CODE2 = 8000
            self.DICT_WORD2 = 8000
        elif dataset == 'medium':
            self.DICT_CODE1 = 1000
            self.DICT_WORD1 = 1000
            self.DICT_CODE2 = 10500
            self.DICT_WORD2 = 10500
        elif dataset == '200':
            self.DICT_CODE1 = 1000
            self.DICT_WORD1 = 1000
            self.DICT_CODE2 = 15500
            self.DICT_WORD2 = 15500
