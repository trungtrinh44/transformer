import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .activations import GELU


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
    def __init__(self, npos, d_model, sinusoid=True):
        """
        npos: number of positions
        d_model: model's dimension size
        """
        super().__init__()
        self.d_model = d_model
        self.sinusoid = sinusoid
        if sinusoid:
            pos = torch.arange(0, npos, 1).float()
            index = torch.arange(0, d_model, 1) // 2 * 2
            index = index.float() / d_model
            index = 10000**index
            pe = pos[:, None] / index[None, :]
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            pe = pe[None, :]
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Parameter(torch.from_numpy(np.random.normal(0., 0.02, (npos, d_model))))

    def forward(self, x):
        """
        x: a tensor of size (batch_size, max_len, d_model)
        """
        return x * (self.d_model ** 0.5) + self.pe[:, :x.size(1)]


def get_seq_mask(lens, device):
    mask = torch.arange(0, lens.max(), 1)[None, :].to(device) >= lens[:, None]
    mask = mask[:, None, None, :]
    return mask


def apply_mask(x, mask):
    x.masked_fill_(mask, float('-inf'))
    return x


def get_look_ahead_mask(max_len, device):
    mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.uint8), 1).to(device)
    mask = mask[None, None, ...]
    return mask


class MultiHeadAtt(nn.Module):
    def __init__(self, nheads, d_model, dropout):
        """
        nheads: number of heads
        d_model: model's dimension size
        """
        assert d_model % nheads == 0, 'd_model must be divisible by nheads'
        super().__init__()
        self.nheads = nheads
        self.d_model = d_model
        self.d_k = d_model // nheads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.att_drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
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
        score = apply_mask(score, mask)
        score = F.softmax(score, dim=-1)
        score = self.att_drop(score)

        outputs = torch.matmul(score, v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.d_model)
        outputs = self.out(outputs)
        outputs = self.drop(outputs)

        return outputs


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class SubLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, outputs, inputs):
        return self.ln(outputs + inputs)


class BasicLayer(nn.Module):
    def __init__(self, d_model, nheads, d_ff, dropout):
        super().__init__()
        self.mult_att = MultiHeadAtt(nheads, d_model, dropout)
        self.sub1 = SubLayer(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub2 = SubLayer(d_model)

    def forward(self, q, k, v, mask):
        x = self.sub1(self.mult_att(q, k, v, mask), q)
        x = self.sub2(self.ff(x), x)
        return x


class EncoderLayer(BasicLayer):
    def __init__(self, d_model, nheads, d_ff, dropout):
        super().__init__(d_model, nheads, d_ff, dropout)

    def forward(self, x, mask):
        return super().forward(x, x, x, mask)


class BasicDecoderLayer(BasicLayer):
    def __init__(self, d_model, nheads, d_ff, dropout):
        super().__init__(d_model, nheads, d_ff, dropout)

    def forward(self, x, mask):
        return super().forward(x, x, x, mask)


class DecoderLayer(BasicLayer):
    def __init__(self, d_model, nheads, d_ff, dropout):
        super().__init__(d_model, nheads, d_ff, dropout)
        self.first_mult_att = MultiHeadAtt(nheads, d_model, dropout)
        self.first_sub = SubLayer(d_model)

    def forward(self, x, mask, enc_input, enc_mask):
        outputs = self.first_sub(self.first_mult_att(x, x, x, mask), x)
        outputs = super().forward(outputs, enc_input, enc_input, enc_mask)
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, nlayers, d_model, nheads, d_ff, vocab_size, npos, dropout, pos_enc_sinusoid=True, layer_output=False):
        super().__init__()
        self.embed = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(npos, d_model, pos_enc_sinusoid)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)
        ])
        self.layer_output = layer_output

    def forward(self, x, lens):
        outputs = self.embed(x)
        outputs = self.pos_enc(outputs)
        outputs = self.dropout(outputs)
        mask = get_seq_mask(lens, x.device)
        if self.layer_output:
            layer_outputs = []
        for layer in self.layers:
            outputs = layer(outputs, mask)
            if self.layer_output:
                layer_outputs.append(outputs)
        if self.layer_output:
            return layer_outputs, mask
        return outputs, mask


class TransformerDecoder(nn.Module):
    def __init__(self, nlayers, d_model, nheads, d_ff, vocab_size, npos, dropout, pos_enc_sinusoid=True):
        super().__init__()
        self.embed = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(npos, d_model, pos_enc_sinusoid)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)
        ])

    def forward(self, x, enc_input, enc_mask):
        outputs = self.embed(x)
        outputs = self.pos_enc(outputs)
        outputs = self.dropout(outputs)
        mask = get_look_ahead_mask(x.size(1), x.device)
        for layer in self.layers:
            outputs = layer(outputs, mask, enc_input, enc_mask)
        return outputs


class TransformerIndependentDecoder(nn.Module):
    def __init__(self, nlayers, d_model, nheads, d_ff, vocab_size, npos, dropout, pos_enc_sinusoid=True):
        super().__init__()
        self.embed = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(npos, d_model, pos_enc_sinusoid)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            BasicDecoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)
        ])

    def forward(self, x):
        outputs = self.embed(x)
        outputs = self.pos_enc(outputs)
        outputs = self.dropout(outputs)
        mask = get_look_ahead_mask(x.size(1), x.device)
        for layer in self.layers:
            outputs = layer(outputs, mask)
        return outputs


class Transformer(nn.Module):
    def __init__(self, nlayers, d_model, nheads, d_ff, src_vocab_size, tgt_vocab_size, npos, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(nlayers, d_model, nheads, d_ff, src_vocab_size, npos, dropout)
        self.decoder = TransformerDecoder(nlayers, d_model, nheads, d_ff, tgt_vocab_size, npos, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, src_lens, tgt):
        outputs, mask = self.encoder(src, src_lens)
        outputs = self.decoder(tgt, outputs, mask)
        outputs = self.out(outputs)
        return outputs
