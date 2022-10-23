import pickle
from utils import get_batch


method = sys.argv[1]
dataset = sys.argv[2]
model_path = sys.argv[3]

if __name__ == "__main__":
    beam_width = 0
    with open("dataset/{}/train/buggy_w2i.pkl".format(dataset), "rb") as f:
        buggy_w2i = pickle.load(f)
    with open("dataset/{}/train/fixed_w2i.pkl".format(dataset), "rb") as f:
        fixed_w2i = pickle.load(f)
    with open("dataset/{}/train/buggy_i2w.pkl".format(dataset), "rb") as f:
        buggy_i2w = pickle.load(f)
    with open("dataset/{}/train/fixed_i2w.pkl".format(dataset), "rb") as f:
        fixed_i2w = pickle.load(f)

    if method == "SmartRep":
        from SmartRep.model import *
        from SmartRep.config import Config
        config = Config(dataset)
        model = Model(config)
        save_path = '{}/save/'.format(method)
        model.load(model_path)
        preds, refs, names = [], [], []
        dir3 = 'dataset/{}/eval/buggy/'.format(dataset)
        dir4 = 'dataset/{}/eval/fixed/'.format(dataset)
        kdir = 'dataset/{}/eval/line3-rep/'.format(dataset)
        dictdir = 'dataset/{}/validation/resultdict/buggy/'.format(dataset)
        real_fix_dir = 'dataset/{}/validation/tokens/fixed/'.format(dataset)

        structurally = 0
        completely = 0
        if beam_width == 0:
            for batch_num, batch in enumerate(get_batch(dir3, kdir, dir4, config, buggy_w2i, fixed_w2i)):
                batch_buggy, batch_keyline, batch_name, batch_fixed = batch
                pred = model((batch_buggy, batch_keyline), False)
                names += batch_name
                preds += pred
                refs += batch_fixed
            for p, r, name in zip(preds, refs, names):
                if p == r:
                    structurally += 1
                with open(real_fix_dir+name) as f:
                    realfix = f.read().replace('\n', ' ').rstrip()
                fix = ' '.join([fixed_i2w[x] for x in p])
                with open(dictdir+name) as df:
                    d = eval(df.read())
                for key in d:
                    fix = fix.replace(key, d[key]['value'])
                if realfix == fix:
                    with open('dataset/{}/validation/tokens/buggy/'.format(dataset) + name) as ff:
                        cf = ff.read().replace('\n',' ').rstrip()
                    completely += 1
            print("{}/{}".format(structurally, len(preds), structurally/len(preds)))
            print("{}/{}".format(completely, len(preds), completely/len(preds)))


    if method == "seq2seq_attn":
        from seq2seq_attn.model import *
        from seq2seq_attn.config import Config
        config = Config(dataset)
        model = Model(config)
        save_path = '{}/save/'.format(method)
        model.load(model_path)
        preds, refs, names = [], [], []
        dir3 = 'dataset/{}/eval/buggy/'.format(dataset)
        dir4 = 'dataset/{}/eval/fixed/'.format(dataset)
        for batch_num, batch in enumerate(get_batch(dir3, None, dir4, config, buggy_w2i, fixed_w2i)):
            batch_buggy, _, batch_name, batch_fixed = batch
            pred = model(batch_buggy, False)
            names += batch_name
            preds += pred
            refs += batch_fixed

        dictdir = 'dataset/{}/validation/resultdict/buggy/'.format(dataset)
        real_fix_dir = 'dataset/{}/validation/tokens/fixed/'.format(dataset)
        structurally = 0
        completely = 0
        for p, r, name in zip(preds, refs, names):
            if p == r:
                structurally += 1
            with open(real_fix_dir+name) as f:
                realfix = f.read().replace('\n', ' ').rstrip()  # 数据集里的fixed
            fix = ' '.join([fixed_i2w[x] for x in p])
            with open(dictdir+name) as df:
                d = eval(df.read())
            for key in d:
                fix = fix.replace(key, d[key]['value'])
            if realfix == fix:
                with open('dataset/{}/validation/tokens/buggy/'.format(dataset) + name) as ff:
                    cf = ff.read().replace('\n',' ').rstrip()
                completely += 1
        print("{}/{}".format(structurally, len(preds), structurally/len(preds)))
        print("{}/{}".format(completely, len(preds), completely/len(preds)))