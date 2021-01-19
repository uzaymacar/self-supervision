"""Vision models for self-supervised pretext tasks."""
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url

# Specify URLs to pretrained models here
MODEL_URLS = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}


class AlexNetBase(nn.Module):
    """
    Implements Base AlexNet with optional batch normalization and adaptive pooling.

    :param (bool) batch_norm: set to True to apply batch normalization
    """
    def __init__(self, batch_norm=True):
        super(AlexNetBase, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                      nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(192) if batch_norm else nn.Identity(),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(384) if batch_norm else nn.Identity(),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Remove any potential nn.Identity() layers
        self.features = nn.Sequential(*[child for child in self.features.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        x = self.features(x)
        if x.shape != (x.shape[0], 256, 6, 6):
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ClassificationHead(nn.Module):
    """
    Implements prediction head for classification tasks with optional batch normalization and
    optional dropout; adaptable with AlexNetBase.

    :param (int) num_classes: number of classes in the classification task
    :param (bool) batch_norm: set to True to apply batch normalization
    :param (float) dropout_rate: dropout rate; set to 0. to disable dropout
    """
    def __init__(self, num_classes=1000, batch_norm=True, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                        nn.Linear(256 * 6 * 6, 4096, bias=False if batch_norm else True),
                                        nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                        nn.Linear(4096, 4096, bias=False if batch_norm else True),
                                        nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, num_classes))

        # Remove any potential nn.Identity() layers
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        x = self.classifier(x)
        return x


class RegressionHead(nn.Module):
    """
    Implements prediction head for regression tasks with optional batch normalization and
    optional dropout; adaptable with AlexNetBase.

    :param (int) output_size: number of values or elements to predict
    :param (bool) batch_norm: set to True to apply batch normalization
    :param (float) dropout_rate: dropout rate; set to 0. to disable dropout
    """
    def __init__(self, output_size=1000, batch_norm=True, dropout_rate=0.5):
        super(RegressionHead, self).__init__()
        self.regressor = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                       nn.Linear(256 * 6 * 6, 4096, bias=False if batch_norm else True),
                                       nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                       nn.Linear(4096, 4096, bias=False if batch_norm else True),
                                       nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(4096, output_size))

        # Remove any potential nn.Identity() layers
        self.regressor = nn.Sequential(*[child for child in self.regressor.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        x = self.regressor(x)
        return x


class DecoderHead(nn.Module):
    """
    Implements a convolutional prediction head for decoding and generation tasks with optional
    batch normalization; adaptable with AlexNetBase.
    NOTE: We are not applying optional dropout as it is typically not used in decoders.

    :param (int) target_size: the target size to resize the output to
    :param (bool) batch_norm: set to True to apply batch normalization
    """
    def __init__(self, target_size, batch_norm=True):
        super(DecoderHead, self).__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(3) if batch_norm else nn.Identity(),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample((target_size, target_size), mode='bilinear', align_corners=True))

        # Remove any potential nn.Identity() layers
        self.decoder = nn.Sequential(*[child for child in self.decoder.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        x = self.decoder(x)
        return x


class AlexNetClassic(nn.Module):
    """
    Implements AlexNet model used for fully-supervised classification task (e.g. ImageNet-1K).
    """
    def __init__(self):
        super(AlexNetClassic, self).__init__()
        self.model = nn.ModuleList([AlexNetBase(batch_norm=True),
                                    ClassificationHead(num_classes=1000,
                                                       batch_norm=True,
                                                       dropout_rate=0.5)])

    def forward(self, x):
        x = self.model[0](x)
        x = self.model[1](x)
        return x


class AlexNetRotation(nn.Module):
    """
    Implements AlexNet model for self-supervised rotation classification task.

    :param (int) num_rotations: number of classes of rotations to use
    """
    def __init__(self, num_rotations=4):
        super(AlexNetRotation, self).__init__()
        self.model = nn.ModuleList([AlexNetBase(batch_norm=True),
                                    ClassificationHead(num_classes=num_rotations,
                                                       batch_norm=True,
                                                       dropout_rate=0.5)])

    def forward(self, x):
        x = self.model[0](x)  # base
        x = self.model[1](x)  # head
        return x


class AlexNetCounting(nn.Module):
    """
    Implements AlexNet model for self-supervised visual counting task.

    :param (int) num_elements: number of individual elements to count for in each image
    """
    def __init__(self, num_elements=1000):
        super(AlexNetCounting, self).__init__()
        self.model = nn.ModuleList([AlexNetBase(batch_norm=True),
                                    RegressionHead(output_size=num_elements,
                                                   batch_norm=True,
                                                   dropout_rate=0.0)])

    def forward(self, x):
        x = self.model[0](x)  # base
        x = self.model[1](x)  # head
        # NOTE: ReLU is applied at the end because we want the counting vector to be all positive
        x = F.relu(x)
        return x


class AlexNetContext(nn.Module):
    """
    Implements AlexNet model for self-supervised context encoder task.

    :param (int) target_size: the target size to resize the output to
    :param (float) dropout_rate: dropout rate; set to 0. to disable dropout
    """
    def __init__(self, target_size, dropout_rate=0.5):
        super(AlexNetContext, self).__init__()
        self.latent_dim = 256 * 6 * 6

        self.encoder = AlexNetBase(batch_norm=True)

        # NOTE: We are using a 1D convolutional layer as the channel-wise fully connected layer.
        self.bottleneck = nn.ModuleList([nn.Conv1d(in_channels=self.latent_dim,
                                                   out_channels=self.latent_dim,
                                                   kernel_size=2,
                                                   groups=self.latent_dim),
                                         nn.Dropout() if dropout_rate > 0. else nn.Identity()])

        self.decoder = DecoderHead(target_size=target_size, batch_norm=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = x.view(-1, 256, 6, 6)
        x = self.decoder(x)
        return x


class AlexNetJigsaw(nn.Module):
    """
    Implements AlexNet model for self-supervised jigsaw puzzle task.

    :param (int) num_tiles: number of tiles to divide each image to for the puzzle
    :param (int) num_permutations: number of permutations to arrange the tiles
    """
    def __init__(self, num_tiles=9, num_permutations=1000):
        super(AlexNetJigsaw, self).__init__()

        # Make sure num. tiles is a number whose square root is an integer (i.e. perfect square)
        assert num_tiles % np.sqrt(num_tiles) == 0
        self.num_tiles = num_tiles

        # TODO: During self-supervised pretraining authors use stride 2 for the first CONV layer
        self.siamese_network = nn.ModuleList([AlexNetBase(batch_norm=True),
                                              nn.Linear(256 * 6 * 6, 512, bias=False),
                                              nn.BatchNorm1d(512),
                                              nn.ReLU(inplace=True)])

        self.classifier = nn.ModuleList([nn.Linear(512 * num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, num_permutations)])

    def forward(self, x):
        assert x.shape[0] == self.num_tiles
        device = x.device

        x = torch.stack([self.siamese_network(tile) for tile in x]).to(device)
        x = x.view(x.shape[1], -1)  # concatenate features from different tiles
        x = self.classifier(x)
        return x


def alexnet_pretrained(progress=True):
    """
    Function for loading pretrained AlexNet model.

    :param (bool) progress: set to True to show the progress bar when downloading the model
    """
    model = AlexNetClassic()
    state_dict = load_state_dict_from_url(MODEL_URLS['alexnet'], progress=progress)
    state_dict_ = deepcopy(state_dict)
    # Rename parameters to include the starting 'model.<index>' string so they match with our models
    for param in state_dict:
        state_dict_['%s%s' % ('model.%d.' % (0 if 'features' in param else 1), param)] = state_dict_.pop(param)
    model.load_state_dict(state_dict_)
    return model
