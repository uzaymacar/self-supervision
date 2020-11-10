import os
import sys
import random
import numpy as np
import torch
import torch.distributed as dist


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


def freeze_weights(model, model_class='alexnet'):
    """
    Freezes all of the weights of the PyTorch model. You have change the last few layers
    AFTER calling this function on your model, otherwise everything will be frozen.
    """
    if model_class == 'alexnet':
        # For AlexNet, below procedure freezes weights up to `conv3`
        for child_no, child in enumerate(model.children()):
            for layer_no, layer in enumerate(child):
                for param in layer.parameters():
                    param.requires_grad = False
                if layer_no == 6:
                    break
            if child_no == 0:
                break

    for param in model.parameters():
        param.requires_grad = False


def setup(rank, world_size):
    """https://pytorch.org/tutorials/intermediate/ddp_tutorial.html"""
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()