import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
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
        return ys,(hx,cx)


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
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.attn = Attn(config)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.dropout3 = nn.Dropout(config.DROPOUT)
        self.embedding = nn.Embedding(config.DICT_WORD, config.EMB_SIZE)
        for i in range(config.NUM_LAYER - 1):
            self.__setattr__("layer_{}".format(i), nn.LSTM(config.DEC_SIZE, config.DEC_SIZE))
        self.lstm = nn.LSTM(2 * config.DEC_SIZE, config.DEC_SIZE)
        self.fc = nn.Linear(config.DEC_SIZE, config.DICT_WORD)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, inputs, l_states, enc, mask):
        config = self.config
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER - 1):
            skip = tensor
            tensor, l_states[i] = getattr(self, "layer_{}".format(i))(tensor, l_states[i])
            tensor = tensor + skip
        context = self.attn(tensor, enc, enc, mask)
        tensor, _ = self.lstm(torch.cat([tensor, context], -1), l_states[-1])
        tensor = self.fc(tensor)
        return tensor

    def get_loss(self, enc, states, targets):
        device = self.device
        config = self.config
        lengths1 = [len(x)+1 for x in targets]
        targets = [torch.tensor([config.START_TOKEN] + x + [config.END_TOKEN]).to(device) for x in targets]
        targets = rnn_utils.pad_sequence(targets)
        inputs = targets[:-1]
        targets = targets[1:]
        lengths2 = [x.shape[0] for x in enc]
        mask = [torch.ones(x).to(device) for x in lengths2]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(max(lengths1), 1, 1)
        mask = mask.eq(1)
        enc = rnn_utils.pad_sequence(enc)
        h, c = states
        enc = self.dropout1(enc)
        dec_hidden = self.dropout2(h)
        dec_cell = self.dropout3(c)
        l_states = [(dec_hidden, dec_cell) for _ in range(config.NUM_LAYER)]
        tensor = self.forward(inputs, l_states, enc, mask)
        loss = 0.
        for i in range(len(lengths1)):
            loss = loss + self.loss_function(tensor[:lengths1[i], i], targets[:lengths1[i], i])
        loss = loss / sum(lengths1)
        return loss

    def translate(self, enc, states):
        device = self.device
        config = self.config
        h, c = states
        lengths = [x.shape[0] for x in enc]
        mask = [torch.ones(x).to(device) for x in lengths]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.eq(1)
        enc = rnn_utils.pad_sequence(enc)
        l_states = [(h, c) for _ in range(config.NUM_LAYER)]
        preds = [[config.START_TOKEN] for _ in range(len(lengths))]
        dec_input = torch.tensor(preds).to(device).view(1, -1)
        for t in range(config.MAX_FIXED_LEN):
            tensor = self.forward(dec_input, l_states, enc, mask)
            dec_input = torch.argmax(tensor, -1)[-1:].detach()
            for i in range(len(lengths)):
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
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.set_trainer()
        self.to(self.device)

    def forward(self, inputs, mode, targets=None):
        if mode:
            return self.train_on_batch(inputs, targets)
        else:
            return self.translate(inputs)

    def train_on_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        self.train()
        enc, (h,c) = self.encoder(inputs)
        loss = self.decoder.get_loss(enc, (h,c), targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def translate(self, inputs):
        self.eval()
        enc, (h, c) = self.encoder(inputs)
        return self.decoder.translate(enc, (h, c))

    def save(self, path):
        checkpoint = {
            'config': self.config,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'optimizer': self.optimizer
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.encoder = checkpoint['encoder']
        self.decoder = checkpoint['decoder']
        self.optimizer = checkpoint['optimizer']

    def set_trainer(self):
        config = self.config
        self.optimizer = optim.Adam(params=[
            {"params": self.parameters()}
        ], lr=config.LR)
