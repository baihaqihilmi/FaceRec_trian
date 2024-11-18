import torch.optim as optim


def get_optimizer(params, cfg):
    if cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'ADAM':
        optimizer = optim.Adam(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    return optimizer