import numpy as np
from torch import nn
from torch.utils.data import Sampler


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
    def __init__(self, data_source, buckets):
        self.data_source = data_source
        self.bucket_len = [0] + buckets + [1e9]
        self.lens = np.array([len(x) for x in data_source], dtype=np.int32).sort()

    def __iter__(self):
        self.buckets = [((self.lens < max_len) & (self.lens >= min_len)).nonzero()[0] for min_len, max_len in zip(self.bucket_len[:-1], self.bucket_len[1:])]
        np.random.shuffle(self.buckets)
        for b in self.buckets:
            np.random.shuffle(b)
        return iter(np.concatenate(self.buckets))

    def __len__(self):
        return len(self.data_source)
