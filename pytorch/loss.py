import torch


def categorical_cross_entropy(pred, gold, eps, weights=None, ignore_index=0):
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = torch.nn.functional.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    if weights is not None:
        loss = loss * weights[gold]
    loss = loss.masked_select(non_pad_mask).mean()  # average later
    return loss


class LayerwiseCategoricalCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, coeff, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.coeff = coeff

    def forward(self, layers, y):
        return sum(coeff * super().forward(layer, y) for layer, coeff in zip(layers, self.coeff))
