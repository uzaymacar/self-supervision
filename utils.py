import random
import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(y_pred, y_true):
    """Function to calculate multiclass accuracy per batch"""
    y_pred_max = torch.argmax(y_pred, dim=-1)
    correct_pred = (y_pred_max == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

