import torch
from torch import nn


class RNNSequenceClassifier(nn.Module):
    def __init__(self, hidden_size, nlayers, vocab_size, wdims, nclasses, wv=None, layer_outputs=False):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, wdims)
        self.hidden_size = hidden_size
        if wv is not None:
            assert wv.shape == (vocab_size, wdims), "Incompatible word embedding shape"
            self.embeds.load_state_dict({'weight': wv})
        self.cells = nn.ModuleList([
            nn.GRU(wdims if i == 0 else hidden_size*2, hidden_size, num_layers=1, bidirectional=True) for i in range(nlayers)
        ])
        self.layer_outputs = layer_outputs
        self.outs = nn.ModuleList([
            nn.Linear(hidden_size*2, nclasses) for _ in range(nlayers)
        ])

    def init_hidden(self, batch_size):
        return torch.zeros((2, batch_size, self.hidden_size))

    def apply_mask(self, x, lens):
        mask = torch.arange(0, x.size(1), 1)[None, :] >= lens[:, None]
        mask = mask[..., None]
        return x.masked_fill_(mask.to(x.device), 0.0)

    def forward(self, inputs, lengths):
        outputs = self.embeds(inputs)
        outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths)
        if self.layer_outputs:
            all_outputs = []
        for cell, out in zip(self.cells, self.outs):
            outputs, _ = cell(outputs, self.init_hidden(inputs.size(1)).to(inputs.device))
            if self.layer_outputs:
                pad_outputs, pad_lengths = nn.utils.rnn.pad_packed_sequence(outputs)
                pad_outputs = pad_outputs.permute(1, 0, 2)
                pad_outputs = out(pad_outputs)
                pad_outputs = self.apply_mask(pad_outputs, pad_lengths)
                all_outputs.append(all_outputs)
        if self.layer_outputs:
            return all_outputs
        return outputs
