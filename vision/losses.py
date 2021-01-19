"""Losses for self-supervised pretext tasks in vision."""
import numpy as np
import torch
import torch.nn.functional as F


def classification_loss(y_hat, y):
    """
    Implements standard classification loss with cross-entropy.

    :param (torch.Tensor) y_hat:
    :param (torch.Tensor) y: ground truth
    :return: (torch.Tensor) loss
    """
    loss = F.cross_entropy(y_hat, y)
    return loss


def reconstruction_loss(y_hat, y, mask=None, reduction='mean'):
    """
    Implements standard pixel-wise regression loss with mean-squared error.

    :param (torch.Tensor) y_hat: prediction
    :param (torch.Tensor) y: ground truth
    :param (torch.Tensor) mask: mask for weighing or removing loss components
    :param (str) reduction: reduction strategy for computing the final loss term
    :return: (torch.Tensor) loss
    """
    loss = F.mse_loss(y_hat, y, reduction=reduction)
    if mask is not None:
        loss *= mask
    return loss


def counting_loss(originals_y_hat, tiles_y_hat, M=10, reduction='mean'):
    """
    Implements the counting loss (reconstruction + contrastive) as shown in the paper.
    The contrastive loss limits the effects of the least effort bias.

    :param (torch.Tensor) originals_y_hat: prediction of AlexNetCounting on original images
    :param (torch.Tensor) tiles_y_hat: prediction of AlexNetCounting on the tiles
    :param (int) M: the constant used in the contrastive loss
    :param (str) reduction: reduction strategy for computing the final loss term
    :return: (torch.Tensor) loss
    """
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
