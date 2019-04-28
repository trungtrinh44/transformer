import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        vocab_size: number of words in the vocabulary
        d_model: model's dimension size
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, npos, d_model):
        """
        npos: number of positions
        d_model: model's dimension size
        """
        super().__init__()
        pos = torch.arange(0, npos, 1).float()
        index = torch.arange(0, d_model, 1) // 2 * 2
        index = index.float() / d_model
        index = 10000**index
        pe = pos[:, None] / index[None, :]
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        self.pe = pe[None, :].to(device)
        self.d_model = d_model

    def forward(self, x):
        """
        x: a tensor of size (batch_size, max_len, d_model)
        """
        return x * (self.d_model ** 0.5) + self.pe[:, x.size(1)]


class SequenceAttMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lens):
        """
        x: a tensor of size (batch_size, max_len, d_model)
        """
        mask = torch.arange(0, x.size(1), 1)[None, :].to(device) >= lens[:, None]
        x[mask] = -1e4
        return x


class LookAheadAttMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lens):
        """
        x: a tensor of size (batch_size, max_len, d_model)
        """
        lens = x.size(1)
        mask = torch.triu(torch.ones(lens, lens, dtype=torch.long), 1).to(device)
        x[:, mask] = -1e4
        return x


class MultiHeadAtt(nn.Module):
    def __init__(self, nheads, d_model, Mask):
        """
        nheads: number of heads
        d_model: model's dimension size
        Mask: either LookAheadAttMask or SequenceAttMask module
        """
        assert d_model % nheads == 0, 'd_model must be divisible by nheads'
        super().__init__()
        self.nheads = nheads
        self.d_model = d_model
        self.d_k = d_model // nheads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mask = Mask()

    def forward(self, Q, K, V, lens):
        """
        Q: tensor of shape (batch_size, max_len, d_model)
        K: tensor of shape (batch_size, max_len, d_model)
        V: tensor of shape (batch_size, max_len, d_model)
        lens: sequence lens for masking
        """
        bs = Q.size(0)

        q = self.q_linear(Q).view(bs, -1, self.nheads, self.d_k).permute(0, 2, 1, 3)
        k = self.q_linear(K).view(bs, -1, self.nheads, self.d_k).permute(0, 2, 3, 1)
        v = self.q_linear(V).view(bs, -1, self.nheads, self.d_k).permute(0, 2, 1, 3)

        score = torch.matmul(q, k) / (self.d_model**0.5)
        score = self.mask(score, lens)
        score = F.softmax(score, dim=-1)

        outputs = torch.matmul(score, v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.d_model)

        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, d_ff, dropout):
        super().__init__()
        self.mult_att = MultiHeadAtt(nheads, d_model, SequenceAttMask)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, lens):
        x = self.ln1(self.dropout(self.mult_att(x, x, x, lens)) + x)
        x = self.ln2(self.dropout(self.ff2(F.relu(self.ff1(x)))) + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, nlayers, d_model, nheads, d_ff, vocab_size, npos, dropout):
        super().__init__()
        self.embed = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(npos, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)
        ])

    def forward(self, x, lens):
        outputs = self.embed(x)
        outputs = self.pos_enc(outputs)
        outputs = self.dropout(outputs)
        for layer in self.layers:
            outputs = layer(outputs, lens)
        return outputs


class TransformerEncoderClassifier(TransformerEncoder):
    def __init__(self, nlayers, d_model, nheads, d_ff, vocab_size, npos, n_classes, dropout):
        super().__init__(nlayers, d_model, nheads, d_ff, vocab_size, npos, dropout)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x, lens):
        outputs = super().forward(x, lens)
        outputs = self.out(outputs[:, 0, :])
        return F.log_softmax(outputs, dim=-1)
