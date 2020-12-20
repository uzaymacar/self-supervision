import numpy as np
import torch
import torch.nn.functional as F


def classification_loss(y_hat, y):
    return F.cross_entropy(y_hat, y)


def reconstruction_loss(y_hat, y, mask=None):
    loss = F.mse_loss(y_hat, y, reduction='mean')
    if mask is not None:
        loss *= mask
    return loss


# TODO: Look at the reduction techniques in F.mse_loss()
def counting_loss(originals_y_hat, tiles_y_hat, M=10, reduction='mean'):
    """Contrastive loss limits the effects of the least effort bias"""
    # Create a shifted batch to compute the contrastive loss betweeen
    batch_size = originals_y_hat.shape[0]
    shift_index = np.random.randint(low=1, high=batch_size-2)
    others_y_hat = torch.cat([originals_y_hat[shift_index:], originals_y_hat[:shift_index]], dim=0)

    tiles_summation = torch.sum(tiles_y_hat, axis=0)
    originals_difference = torch.abs(originals_y_hat - tiles_summation) ** 2
    contrastive_difference = F.relu(M - (torch.abs(others_y_hat - tiles_summation) ** 2))
    # NOTE: The ReLU here implements a differentiable version of the max{0, M - |...|}
    #       mentioned in the paper in equation (4)

    loss = originals_difference + contrastive_difference
    if reduction is not None:
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)

    return loss
