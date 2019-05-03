from torch import nn


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
