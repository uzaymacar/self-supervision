import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_loss(y_hat, y):
    return F.cross_entropy(y_hat, y)


# TODO: Look at the reduction techniques in F.mse_loss()
def counting_loss(originals_y_hat, tiles_y_hat, others_y_hat, M=10, reduction='mean'):
    """Contrastive loss limits teh effects of the least effort bias"""
    originals_difference = torch.abs(originals_y_hat - torch.sum(tiles_y_hat, axis=0)) ** 2
    contrastive_difference = F.relu(M - torch.abs(others_y_hat - torch.sum(tiles_y_hat)) ** 2)
    # NOTE: The ReLU here implements a differentiable version of the max{0, M - |...|}
    #       mentioned in the paper in equation (4)

    loss = originals_difference + contrastive_difference
    if reduction is not None:
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)

    return loss
