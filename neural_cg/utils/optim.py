from torch import optim
from torch.optim import lr_scheduler

def create_optimizer(parameters, name, params):
    if name == "adam":
        return optim.Adam(parameters, **params)
    elif name == "adamw":
        return optim.AdamW(parameters, **params)
    elif name == "sgd":
        return optim.SGD(parameters, **params)
    else:
        raise ValueError(f"Unknown optimizer {name}")

def create_scheduler(optimizer, name, params):
    if name == "onecycle":
        return lr_scheduler.OneCycleLR(optimizer, **params)
    elif name == "exp":
        return lr_scheduler.ExponentialLR(optimizer, **params)
    elif name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == "cosine_warmup":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler {name}")
