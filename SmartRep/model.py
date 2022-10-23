import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

cuda_choose = "cuda:0"

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(cuda_choose) if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding = nn.Embedding(config.DICT_CODE, config.EMB_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i), nn.LSTM(config.ENC_SIZE, config.ENC_SIZE))

    def forward(self, inputs):
        device = self.device
        config = self.config
        lengths = [len(x) for x in inputs]
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = rnn_utils.pad_sequence(inputs)
        tensor = self.embedding(inputs + 1)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor, lengths, enforce_sorted=False)
            tensor, (h,c) = getattr(self, "layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        cx = c
        hx = h
        ys = [y[:i] for y, i in zip(torch.unbind(tensor, axis=1), lengths)]
        return ys, (hx,cx)


class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Q = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.K = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.V = nn.Linear(config.ENC_SIZE, config.ATTN_SIZE)
        self.W = nn.Linear(config.ATTN_SIZE, 1)

    def forward(self, q, k, v, mask):
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q + k))
        attn_weight = attn_weight.squeeze(-1)
        attn_weight = torch.where(mask, attn_weight, torch.tensor(-1e6).to(q.device))
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight * v
        context = context.sum(1)
        return context


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(cuda_choose) if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.attn = Attn(config)
        self.attn2 = Attn(config)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.dropout3 = nn.Dropout(config.DROPOUT)
        self.dropout4 = nn.Dropout(config.DROPOUT)
        self.dropout5 = nn.Dropout(config.DROPOUT)
        self.embedding = nn.Embedding(config.DICT_WORD, config.EMB_SIZE)
        self.linear1 = nn.Linear(2 * config.DEC_SIZE, self.config.DEC_SIZE)
        self.linear2 = nn.Linear(2 * config.DEC_SIZE, self.config.DEC_SIZE)
        for i in range(config.NUM_LAYER - 1):
            self.__setattr__("layer_{}".format(i), nn.LSTM(config.DEC_SIZE, config.DEC_SIZE))
        self.lstm = nn.LSTM(2 * config.DEC_SIZE, config.DEC_SIZE)
        self.fc = nn.Linear(config.DEC_SIZE, config.DICT_WORD)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, inputs, l_states, enc1, enc2, mask1, mask2):
        config = self.config
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER - 1):
            skip = tensor
            tensor, l_states[i] = getattr(self, "layer_{}".format(i))(tensor, l_states[i])
            tensor = tensor + skip
        context1 = self.attn(tensor, enc1, enc1, mask1)
        context2 = self.attn2(tensor, enc2, enc2, mask2)
        context = context1 + context2
        tensor, l_states[-1] = self.lstm(torch.cat([tensor, context], -1), l_states[-1])
        tensor = self.fc(tensor)
        return tensor, l_states

    def get_loss(self, enc1, hc1, enc2, hc2, targets):
        device = self.device
        config = self.config
        leng = [len(x) + 1 for x in targets]
        targets = [torch.tensor([config.START_TOKEN] + x + [config.END_TOKEN]).to(device) for x in targets]
        targets = rnn_utils.pad_sequence(targets)
        inputs = targets[:-1]
        targets = targets[1:]

        lengths1 = [x.shape[0] for x in enc1]
        mask1 = [torch.ones(x).to(device) for x in lengths1]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.repeat(max(leng), 1, 1)
        mask1 = mask1.eq(1)

        lengths2 = [x.shape[0] for x in enc2]
        mask2 = [torch.ones(x).to(device) for x in lengths2]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.repeat(max(leng), 1, 1)
        mask2 = mask2.eq(1)

        enc1 = rnn_utils.pad_sequence(enc1)
        enc1 = self.dropout1(enc1)
        enc2 = rnn_utils.pad_sequence(enc2)
        enc2 = self.dropout2(enc2)

        h1, c1 = hc1
        h2, c2 = hc2
        h12 = torch.cat([h1, h2], -1)
        h12 = self.linear1(h12)
        h12 = F.relu(h12)
        c12 = torch.cat([c1, c2], -1)
        c12 = self.linear2(c12)
        c12 = F.relu(c12)

        h12 = self.dropout3(h12)
        c12 = self.dropout4(c12)
        hc12 = [(h12,c12) for _ in range(config.NUM_LAYER)]
        tensor, _ = self.forward(inputs, hc12, enc1, enc2, mask1, mask2)

        batch_size = len(leng)
        loss = 0.
        for i in range(batch_size):
            loss = loss + self.loss_function(tensor[:leng[i],i], targets[:leng[i],i])
        loss = loss / sum(leng)
        return loss


    def beam(self, inputs, hc12, enc1, enc2, size):
        config = self.config

        lengths1_enc = [x.shape[0] for x in enc1]
        enc1 = rnn_utils.pad_sequence(enc1)
        mask1 = [torch.ones(x).to(enc1.device) for x in lengths1_enc]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)

        lengths2_enc = [x.shape[0] for x in enc2]
        enc2 = rnn_utils.pad_sequence(enc2)
        mask2 = [torch.ones(x).to(enc2.device) for x in lengths2_enc]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)

        enc1 = enc1.unsqueeze(2).repeat(1, 1, size, 1).view(enc1.shape[0], -1, enc1.shape[-1])
        enc2 = enc2.unsqueeze(2).repeat(1, 1, size, 1).view(enc2.shape[0], -1, enc2.shape[-1])
        mask1 = mask1.unsqueeze(-1).repeat(1, 1, 1, size).view(mask1.shape[0], mask1.shape[1], -1)
        mask2 = mask2.unsqueeze(-1).repeat(1, 1, 1, size).view(mask2.shape[0], mask2.shape[1], -1)
        h, c = hc12[0]
        h = h.unsqueeze(2).repeat(1, 1, size, 1).view(h.shape[0], -1, h.shape[-1])
        c = c.unsqueeze(2).repeat(1, 1, size, 1).view(c.shape[0], -1, c.shape[-1])
        hc12 = [(h, c) for _ in range(config.NUM_LAYER + 1)]
        lengths_input = [x.shape[0] for x in inputs]
        tensor = rnn_utils.pad_sequence(inputs)
        for i in range(config.NUM_LAYER - 1):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
            tensor, hc12[i] = getattr(self, "layer_{}".format(i))(tensor, hc12[i])
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        # with torch.no_grad():
        context1 = self.attn(tensor, enc1, enc1, mask1)
        context2 = self.attn2(tensor, enc2, enc2, mask2)
        context = context1 + context2
        tensor = torch.cat([tensor, context], -1)
        tensor = rnn_utils.pack_padded_sequence(tensor, lengths_input, enforce_sorted=False)
        tensor, hc12[-1] = self.lstm(tensor, hc12[-1])
        tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        return tensor, hc12


    def beam_search(self, enc1, enc2, hc1, hc2, size):
        with torch.no_grad():
            device = self.device
            config = self.config

            lengths1 = [x.shape[0] for x in enc1]
            mask1 = [torch.ones(x).to(device) for x in lengths1]
            mask1 = rnn_utils.pad_sequence(mask1)
            mask1 = mask1.unsqueeze(0)
            mask1 = mask1.eq(1)

            lengths2 = [x.shape[0] for x in enc2]

            mask2 = [torch.ones(x).to(device) for x in lengths2]
            mask2 = rnn_utils.pad_sequence(mask2)
            mask2 = mask2.unsqueeze(0)
            mask2 = mask2.eq(1)

            h1, c1 = hc1
            h2, c2 = hc2
            h12 = torch.cat([h1, h2], -1)
            h12 = self.linear1(h12)
            h12 = F.relu(h12)
            c12 = torch.cat([c1, c2], -1)
            c12 = self.linear2(c12)
            c12 = F.relu(c12)
            hc12 = [(h12, c12) for _ in range(config.NUM_LAYER + 1)]

            batch_size = config.BATCH_SIZE
            start_token = config.START_TOKEN
            end_token = config.END_TOKEN
            tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
            tensor = tensor.view(1, -1)

            new_enc1 = rnn_utils.pad_sequence(enc1)
            new_enc2 = rnn_utils.pad_sequence(enc2)
            tensor, hc12 = self.forward(tensor, hc12, new_enc1, new_enc2, mask1, mask2)

            tensor = tensor[:,:,:-1]
            tensor = tensor[-1].softmax(-1).topk(size, -1)
            probs = torch.log(tensor.values).view(-1, 1)
            indices = tensor.indices.view(-1, 1)
            preds = [[config.START_TOKEN, int(indices[i][0])] for i in range(indices.shape[0])]
            completed = [([], 0) for _ in lengths1]
            tensor = torch.tensor([preds[j][-1] for j in range(len(preds))]).to(device)
            for i in range(config.MAX_FIXED_LEN - 1):
                tensor = tensor.view(1, -1)
                tensor = self.embedding(tensor)
                tensor = tensor.unbind(1)
                tensor, hc12 = self.beam(tensor, hc12, enc1, enc2, size)
                tensor = self.fc(tensor)
                tensor = tensor[-1].softmax(-1).topk(size, -1)
                probs = probs + torch.log(tensor.values)
                probs = probs.view(-1, size * size)
                indices = tensor.indices.view(-1, size * size)
                probs = probs.topk(size, -1)
                tensor = probs.indices
                probs = probs.values.view(-1, 1)
                _preds = []
                for j in range(batch_size):
                    for k in range(j * size, j * size + size):
                        index = k - j * size
                        index = int(tensor[j][index])
                        l = j * size + index // size
                        pred = [x for x in preds[l]]
                        if pred[-1] != end_token:
                            pred += [int(indices[j][index])]
                            if pred[-1] == end_token:
                                if len(completed[j][0]) == 0 or completed[j][1] < float(probs[k]):
                                    completed[j] = (pred, float(probs[k]))
                        _preds.append(pred)
                preds = _preds
                tensor = torch.tensor([preds[x][-1] for x in range(len(preds))]).to(device)
            _preds = [preds[j * size] if len(completed[j][0]) == 0 or probs[j * size] > completed[j][1] else completed[j][0] for j in range(len(lengths1))]
            preds = [x[1:-1] if x[-1] == config.END_TOKEN else x[1:] for x in preds]
            _preds = [x[1:-1] if x[-1] == config.END_TOKEN else x[1:] for x in _preds]
            return preds, _preds


    def translate(self, enc1, enc2, hc1, hc2):
        device = self.device
        config = self.config

        lengths1 = [x.shape[0] for x in enc1]
        enc1 = rnn_utils.pad_sequence(enc1)
        mask1 = [torch.ones(x).to(device) for x in lengths1]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)

        lengths2 = [x.shape[0] for x in enc2]
        enc2 = rnn_utils.pad_sequence(enc2)
        mask2 = [torch.ones(x).to(device) for x in lengths2]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)

        h1, c1 = hc1
        h2, c2 = hc2
        h12 = torch.cat([h1,h2], -1)
        h12 = self.linear1(h12)
        h12 = F.relu(h12)
        c12 = torch.cat([c1,c2], -1)
        c12 = self.linear2(c12)
        c12 = F.relu(c12)
        hc12 = [(h12,c12) for _ in range(config.NUM_LAYER)]

        batch_size = len(lengths1)
        preds = [[config.START_TOKEN] for _ in range(len(lengths1))]
        dec_input = torch.tensor(preds).to(device).view(1,-1)
        for _ in range(config.MAX_FIXED_LEN):
            tensor = self.forward(dec_input, hc12, enc1, enc2, mask1, mask2)
            dec_input = torch.argmax(tensor, -1)[-1:].detach()
            for i in range(batch_size):
                if preds[i][-1] != config.END_TOKEN:
                    preds[i].append(int(dec_input[0, i]))
        preds = [x[1:-1] if x[-1] == config.END_TOKEN else x[1:] for x in preds]

        return preds


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.SEED) 
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device(cuda_choose) if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.encoder1 = Encoder(config)
        self.encoder2 = Encoder(config)
        self.decoder = Decoder(config)
        self.set_trainer()
        self.to(self.device)

    def forward(self, inputs, mode, targets=None, size=-1):
        if mode:
            return self.train_on_batch(inputs, targets)
        elif size == 0:
            return self.translate(inputs)
        elif size == -1:
            raise Exception
        else:  # size>0
            return self.beam_search(inputs, size)

    def beam_search(self, inputs, size):
        self.eval()
        end_token = self.config.END_TOKEN
        enc1, (h1,c1) = self.encoder1(inputs[0])
        enc2, (h2,c2) = self.encoder2(inputs[1])
        preds = self.decoder.beam_search(enc1, enc2, (h1,c1), (h2,c2), size)
        preds = [x[:-1] if x[-1] == end_token else x for x in preds]
        return preds

    def train_on_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        self.train()
        enc1, (h1,c1) = self.encoder1(inputs[0])
        enc2, (h2,c2) = self.encoder2(inputs[1])

        loss = self.decoder.get_loss(enc1, (h1,c1), enc2, (h2,c2), targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def translate(self, inputs):
        self.eval()
        enc1, (h1,c1) = self.encoder1(inputs[0])
        enc2, (h2,c2) = self.encoder2(inputs[1])
        _preds = self.decoder.translate(enc1, enc2, (h1,c1), (h2,c2))
        return _preds

    def save(self, path):
        checkpoint = {
            'config': self.config,
            'encoder1': self.encoder1,
            'encoder2': self.encoder2,
            'decoder': self.decoder,
            'optimizer': self.optimizer
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.encoder1 = checkpoint['encoder1']
        self.encoder2 = checkpoint['encoder2']
        self.decoder = checkpoint['decoder']
        self.optimizer = checkpoint['optimizer']

    def set_trainer(self):
        config = self.config
        self.optimizer = optim.Adam(params=[
            {"params": self.parameters()}
        ], lr=config.LR)
