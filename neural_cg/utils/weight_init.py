from torch import nn
def weight_init(m: nn.Module):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
