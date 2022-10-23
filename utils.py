import os
from nltk.translate.bleu_score import *
import logging


def get_time(t):
    t = int(t)
    h = t // 3600
    m = t % 3600 // 60
    s = t % 60
    h = str(h)
    if m < 10:
        m = '0' + str(m)
    else:
        m = str(m)
    if s < 10:
        s = '0' + str(s)
    else:
        s = str(s)
    return h + ":" + m + ":" + s


def get_one(d1, d2, config, in_dir1, in_dir2, out_dir):
    for name in os.listdir(in_dir1):
        path1 = in_dir1 + name
        path3 = out_dir + name
        with open(path1) as f1, open(path3) as f3:
            c1, c3 = f1.read(), f3.read()
        bugcode_token_list = c1.split(' ')
        fixcode_token_list = c3.split(' ')
        in1 = [d1[x] if x in d1 else d1['<UNK>'] for x in bugcode_token_list]
        out = [d2[x] if x in d2 else d2['<UNK>'] for x in fixcode_token_list]
        in1 = in1[:config.MAX_BUGGY_LEN]
        out = out[:config.MAX_BUGGY_LEN]
        in2 = []
        if in_dir2 is not None:
            path2 = in_dir2 + name
            with open(path2) as f2:
                c2 = f2.read()
            keyline_token_list = c2.split(' ')
            in2 = [d1[x] if x in d1 else d1['<UNK>'] for x in keyline_token_list]
            in2 = in2[:config.MAX_BUGGY_LEN]
        yield in1, in2, name, out


def get_batch(in_dir1, in_dir2, out_dir, conf, d1, d2):
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    batch_size = conf.BATCH_SIZE
    for bugcode, keyline, contract_name, fixcode in get_one(d1, d2, conf, in_dir1, in_dir2=in_dir2, out_dir=out_dir):
        batch_in1.append(bugcode)
        batch_in2.append(keyline)  # keyline maybe []
        batch_in3.append(contract_name)
        batch_out.append(fixcode)
        if len(batch_in1) >= batch_size:
            yield batch_in1, batch_in2, batch_in3, batch_out
            batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_in1) > 0:
        yield batch_in1, batch_in2, batch_in3, batch_out


def get_one3(d1, d2, d3, config, in_dir1, in_dir2, out_dir):
    for name in os.listdir(in_dir1):
        path1 = in_dir1 + name
        path3 = out_dir + name
        with open(path1) as f1, open(path3) as f3:
            c1, c3 = f1.read(), f3.read()
        bugcode_token_list = c1.split(' ')
        fixcode_token_list = c3.split(' ')
        in1 = [d1[x] if x in d1 else d1['<UNK>'] for x in bugcode_token_list]
        out = [d2[x] if x in d2 else d2['<UNK>'] for x in fixcode_token_list]
        in1 = in1[:config.MAX_BUGGY_LEN]
        out = out[:config.MAX_BUGGY_LEN]
        in2 = []
        if in_dir2 is not None:
            path2 = in_dir2 + name
            with open(path2) as f2:
                c2 = f2.read()
            keyline_token_list = c2.split(' ')
            in2 = [d3[x] if x in d3 else d3['<UNK>'] for x in keyline_token_list]
            in2 = in2[:config.MAX_BUGGY_LEN]
        yield in1, in2, name, out


def get_batch3(in_dir1, in_dir2, out_dir, conf, d1, d2, d3):
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    batch_size = conf.BATCH_SIZE
    for bugcode, keyline, contract_name, fixcode in get_one3(d1, d2, d3, conf, in_dir1, in_dir2=in_dir2, out_dir=out_dir):
        batch_in1.append(bugcode)
        batch_in2.append(keyline)  # keyline maybe []
        batch_in3.append(contract_name)
        batch_out.append(fixcode)
        if len(batch_in1) >= batch_size:
            yield batch_in1, batch_in2, batch_in3, batch_out
            batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_in1) > 0:
        yield batch_in1, batch_in2, batch_in3, batch_out


def set_logger(dataset, method):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    log_filename = "dataset/{}/{}_{}.log".format(dataset, dataset, method)
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
