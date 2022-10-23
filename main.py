import os, random
import pickle
from time import time
from utils import get_batch, set_logger, get_time
import sys


method = sys.argv[1]
dataset = sys.argv[2]
start = sys.argv[3]

if __name__ == "__main__":
    logger = set_logger(dataset, method)

    with open("dataset/{}/train/buggy_w2i.pkl".format(dataset), "rb") as f:
        buggy_w2i = pickle.load(f)
    with open("dataset/{}/train/fixed_w2i.pkl".format(dataset), "rb") as f:
        fixed_w2i = pickle.load(f)


    if method == "seq2seq_attn":
        from seq2seq_attn.model import *
        from seq2seq_attn.config import Config
        config = Config(dataset)
        model = Model(config)
        save_path = '{}/save/'.format(method)
        os.makedirs(save_path, exist_ok=True)
        random.seed(config.SEED)

        if start != -1:
            model.load(save_path + "{}_{}.pkl".format(dataset, start))
            model.set_trainer()

        t0 = time()
        best_sames = 0
        for epoch in range(start+1, config.EPOCH):
            loss = []
            dir1 = 'dataset/{}/train/buggy/'.format(dataset)
            dir2 = 'dataset/{}/train/fixed/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(dir1,None,dir2,config,buggy_w2i,fixed_w2i)):
                batch_buggy, _, _, batch_fixed = batch
                loss = model(batch_buggy, True, batch_fixed)
                logger.info("Epoch={}/{}, Batch={}, train_loss={}, time={}".format(epoch, config.EPOCH, batch_num, loss, get_time(time()-t0)))

            preds, refs = [], []
            dir3 = 'dataset/{}/eval/buggy/'.format(dataset)
            dir4 = 'dataset/{}/eval/fixed/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(dir3,None,dir4,config,buggy_w2i,fixed_w2i)):
                batch_buggy, _, _, batch_fixed = batch
                pred = model(batch_buggy, False)
                preds += pred
                refs += batch_fixed

            sames = 0
            for p, r in zip(preds, refs):
                if p == r:
                    sames += 1
            if sames > best_sames:
                best_sames = sames
                model.save(save_path + "{}_{}.pkl".format(dataset, epoch))
                logger.info("Saving the model {} with the best same result {}/{}\n".format(epoch, best_sames, len(preds)))  # 取最优的保存


    if method == "SmartRep":
        from SmartRep.model import *
        from SmartRep.config import Config
        config = Config(dataset)
        model = Model(config)
        save_path = '{}/save/'.format(method)
        os.makedirs(save_path, exist_ok=True)
        random.seed(config.SEED)

        if start != -1:
            model.load(save_path + "{}_{}.pkl".format(dataset, start))  # 如果模型有的话就读取模型
            model.set_trainer()

        t0 = time()
        best_sames = 0
        for epoch in range(start+1, config.EPOCH):
            loss = []
            dir1 = 'dataset/{}/train/buggy/'.format(dataset)
            dir2 = 'dataset/{}/train/fixed/'.format(dataset)
            kdir = 'dataset/{}/train/line3-rep/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(dir1, kdir, dir2, config, buggy_w2i, fixed_w2i)):
                batch_buggy, batch_keyline, _, batch_fixed = batch
                loss = model((batch_buggy, batch_keyline), True, batch_fixed)
                logger.info("Epoch={}/{}, Batch={}, train_loss={}, time={}".format(epoch, config.EPOCH, batch_num, loss, get_time(time()-t0)))

            preds, refs = [], []
            vdir1 = 'dataset/{}/eval/buggy/'.format(dataset)
            vdir2 = 'dataset/{}/eval/fixed/'.format(dataset)
            kdir2 = 'dataset/{}/eval/line3-rep/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(vdir1, kdir2, vdir2, config, buggy_w2i, fixed_w2i)):
                batch_buggy, batch_keyline, _, batch_fixed = batch
                pred = model((batch_buggy, batch_keyline), False)
                preds += pred
                refs += batch_fixed

            sames = 0
            for p, r in zip(preds, refs):
                if p == r:
                    sames += 1
            if sames > best_sames:
                best_sames = sames
                model.save(save_path + "{}_{}.pkl".format(dataset, epoch))
                logger.info("Saving the model {} with the best same result {}/{}\n".format(epoch, best_sames, len(preds)))  # 取最优的保存


    if method == "seq2seq":
        from seq2seq.model import *
        from seq2seq.config import Config
        config = Config(dataset)
        model = Model(config)
        save_path = '{}/save/'.format(method)
        os.makedirs(save_path, exist_ok=True)
        random.seed(config.SEED)

        if start != -1:
            model.load(save_path + "{}_{}.pkl".format(dataset, start))
            model.set_trainer()

        t0 = time()
        best_sames = 0
        for epoch in range(start+1, config.EPOCH):
            loss = []
            dir1 = 'dataset/{}/train/buggy/'.format(dataset)
            dir2 = 'dataset/{}/train/fixed/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(dir1, None, dir2, config, buggy_w2i, fixed_w2i)):
                batch_buggy, _, _, batch_fixed = batch
                loss = model(batch_buggy, True, batch_fixed)
                logger.info("Epoch={}/{}, Batch={}, train_loss={}, time={}".format(epoch, config.EPOCH, batch_num, loss, get_time(time()-t0)))

            preds, refs = [], []
            dir3 = 'dataset/{}/eval/buggy/'.format(dataset)
            dir4 = 'dataset/{}/eval/fixed/'.format(dataset)
            for batch_num,batch in enumerate(get_batch(dir3, None, dir4, config, buggy_w2i, fixed_w2i)):
                batch_buggy, _, _, batch_fixed = batch
                pred = model(batch_buggy, False)
                preds += pred
                refs += batch_fixed

            sames = 0
            for p, r in zip(preds, refs):
                if p == r:
                    sames += 1
            if sames > best_sames:
                best_sames = sames
                model.save(save_path + "{}_{}.pkl".format(dataset, epoch))
                logger.info("Saving the model {} with the best same result {}/{}\n".format(epoch, best_sames, len(preds)))  # 取最优的保存
