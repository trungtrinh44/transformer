#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unicodedata
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import numpy as np
import re
import torch
from torch import nn, functional as F, optim
import time
import sentencepiece as spm
from tqdm import tqdm
from apex import amp
from transformer.pytorch.transformer import TransformerIndependentDecoder, get_seq_mask, apply_mask
from transformer.pytorch.utils import init_transformer, BucketByLengthSampler
# In[2]:


sp = spm.SentencePieceProcessor()
sp.Load("m.model")


# In[3]:


# with open('data2id.txt', 'r') as fin:
#     data = [x.strip() for x in tqdm(fin)]
data=[]
with open('data/train_id.txt') as fin:
    data.extend(x.strip().split() for x in tqdm(fin))
# with open('data/vi/target_train.txt') as fin:
#     data.extend(x.strip() for x in tqdm(fin))
print(len(data))


# In[4]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


# In[5]:


class TextDataset(Dataset):
    def __init__(self, X):
        self.X = X
#         self.cache = {}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
#         cache = self.cache
#         if idx not in cache:
#             cache[idx] = sp.EncodeAsIds(self.X[idx])
#         return cache[idx]
#         return sp.EncodeAsIds(self.X[idx])
        return self.X[idx]


# In[6]:


def get_random_snippet(s, maxlen):
    if len(s) <= maxlen:
        return s
    start = np.random.randint(0, len(s)-maxlen)
    return s[start:start+maxlen]


# In[7]:


class Padder(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        
    def padding(self, x):
        seq_lens = np.array([len(y) for y in x], dtype=np.int64)
        out = np.zeros((len(x), max(seq_lens)), dtype=np.int64)
        for ind, y in zip(out, x):
            ind[:len(y)]=y
        return torch.from_numpy(out), torch.from_numpy(seq_lens)
    
    def __call__(self, tensors):
        maxlen = self.maxlen
#         bptt = maxlen if np.random.random() < 0.90 else maxlen / 2.
#         seq_len = max(10, int(np.random.normal(bptt, 10)))
        seq_len = maxlen
        texts = [x[:seq_len] for x in tensors]
        return self.padding(texts)


# In[8]:


def get_data_loader(X, bs, shuffle, padder):
    ds = TextDataset(X)
    dl = DataLoader(ds, bs, collate_fn=padder, num_workers=4, sampler=BucketByLengthSampler(ds, [64, 128, 256]))
    return dl


# In[9]:


# dl = get_data_loader(data, 2, False, 256, 0.5)


# In[10]:


# idx2char={i: c for c,i in char2idx.items()}


# In[11]:


class LinearWarmupRsqrtDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_step, last_epoch=-1):
        self.warmup_step = warmup_step
        super(LinearWarmupRsqrtDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min((max(1.0, self.last_epoch)**-0.1)*(self.warmup_step**0.1), self.last_epoch/self.warmup_step)
                for base_lr in self.base_lrs]


# In[12]:




# In[13]:


model = TransformerIndependentDecoder(12, 512, 16, 512*4, len(sp), 512, 0.1, False)
decoder = nn.Linear(512, len(sp))


# In[ ]:


model.to(device)
decoder.to(device)

for m in model.modules():
    init_transformer(m)
for m in decoder.modules():
    init_transformer(m)
# Tie weights

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
BATCH_SIZE=4
COUNT=8
EPOCH=50
lr0 = 0.25
lr1 = 1e-5
TOTAL_ITER = ((len(data)-1)//BATCH_SIZE + 1)*EPOCH
optimizer = optim.SGD(model.parameters(), lr0)
scheduler = LinearWarmupRsqrtDecayScheduler(optimizer, 5000) #(len(data)-1)//BATCH_SIZE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, (len(data)-1)//BATCH_SIZE+1, 0.5)
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# In[ ]:


def clip_by_norm(value, max_val):
    n = torch.norm(value)
#     print(n)
    return value/n * min(n, max_val)


# In[ ]:


# for p in model.parameters():
#     p.register_hook(lambda grad: clip_by_norm(grad, 1.0))


# In[ ]:


def accuracy(out, label):
    eq = (out==label)
    not_mask = (label != 0)
    return eq.masked_select(not_mask).float().sum(), not_mask.float().sum()


# In[ ]:


# get_ipython().system('mkdir -p language_model')


# In[ ]:


model.train()
torch.cuda.empty_cache()
padder = Padder(512)
SAVE_FREQ = ((len(data)-1)//BATCH_SIZE+1)//10
train_iter = get_data_loader(data, BATCH_SIZE, True, padder)

def batch_iter(child_iter, batch_size):
    result = []
    for item in child_iter:
        if len(result) == batch_size:
            yield result
            result = []
        result.append(item)
    
optimizer.zero_grad()
for _ in range(EPOCH):
    for items in batch_iter(train_iter, COUNT):
        t0 = time.time()
        all_loss = 0.0
        acc = 0.0
        nz_count = sum(x.sum() for _, x in items)
        for tgt, tgt_lens in items:
            tgt = tgt.to(device)
            tgt_lens = tgt_lens.to(device)
            outputs = model(tgt[:,:-1])
            outputs = decoder(outputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), tgt[:,1:].contiguous().view(-1)) / nz_count
            loss.backward()
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()
            a1, _ = accuracy(outputs.argmax(2), tgt[:,1:])
            acc += a1 / nz_count
            all_loss += loss
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print('\rIter {:06d}, ntokens {:06d}, loss: {:.05f}, acc: {:.05f}, lr: {:.05f}, time: {:.05f}'.format(scheduler.last_epoch, nz_count, all_loss, acc, optimizer.param_groups[0]['lr'], time.time()-t0), end='')
        t0 = time.time()
        if (scheduler.last_epoch +1) % SAVE_FREQ == 0:
            torch.save(model.state_dict(), 'language_model/model16.pt.{}'.format(scheduler.last_epoch))
            torch.save(decoder.state_dict(), 'language_model/decoder16.pt.{}'.format(scheduler.last_epoch))
    torch.save(model.state_dict(), 'language_model/model16.pt.{}'.format(scheduler.last_epoch))
    torch.save(decoder.state_dict(), 'language_model/decoder16.pt.{}'.format(scheduler.last_epoch))





