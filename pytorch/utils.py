import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler, Dataset


def init_transformer(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m, 0., 0.02)
    elif classname.find('TransformerXL') != -1:
        if hasattr(m, 'u'):
            nn.init.normal_(m.u, 0., 0.02)
        if hasattr(m, 'v'):
            nn.init.normal_(m.v, 0., 0.02)


class BucketByLengthSampler(Sampler):
    def __init__(self, data_source, buckets, batch_size, maxlen):
        self.data_source = data_source
        self.bucket_len = [0] + buckets + [1e9]
        self.lens = np.array([len(x) for x in data_source], dtype=np.int32)
        self.batch_size = batch_size

    def __iter__(self):
        self.buckets = [((self.lens < max_len) & (self.lens >= min_len)).nonzero()[0] for min_len, max_len in zip(self.bucket_len[:-1], self.bucket_len[1:])]
        np.random.shuffle(self.buckets)
        for b in self.buckets:
            np.random.shuffle(b)
        items = np.concatenate(self.buckets)
        nbatch = (len(self.data_source)-1)//self.batch_size + 1
        if len(items) < nbatch * self.batch_size:
            items = np.concatenate([items, np.zeros((nbatch*self.batch_size-len(items),), np.int32)-1])
        items = np.reshape(items, (nbatch, self.batch_size))
        np.random.shuffle(items)
        items = items.flatten()
        items = items[items >= 0]
        for idx in range(0, len(items), self.batch_size):
            yield items[idx:idx+self.batch_size]

    def __len__(self):
        return len(self.data_source)


class Padder(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def padding(self, x):
        seq_lens = np.array([len(y) for y in x], dtype=np.int64)
        out = np.zeros((len(x), max(seq_lens)), dtype=np.int64)
        for ind, y in zip(out, x):
            ind[:len(y)] = y
        return torch.from_numpy(out), torch.from_numpy(seq_lens)

    def __call__(self, tensors):
        maxlen = self.maxlen
        seq_len = maxlen
        texts = [x[:seq_len] for x in tensors]
        return self.padding(texts)


class TextDataset(Dataset):
    def __init__(self, X, maxlen):
        self.maxlen = maxlen
        self.X = [s for x in X for s in self.__get_snippets(x)]

    def __get_snippets(self, text):
        return [
            text[i:i+self.maxlen] for i in range(0, len(text), self.maxlen)
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
