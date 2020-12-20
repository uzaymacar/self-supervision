from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url

# Specify URLs to pretrained models here
MODEL_URLS = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetBase(nn.Module):
    """
    Implements Base AlexNet with optional batch normalization and adaptive pooling.
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
    batch normalization and optional dropout; adaptable with AlexNetBase.
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
    NOTE: We are using a 1D convolutional layer as the channel-wise fully connected layer.
    """
    def __init__(self, target_size, dropout_rate=0.5):
        super(AlexNetContext, self).__init__()
        num_channels = 256 * 6 * 6
        self.model = nn.ModuleList([AlexNetBase(batch_norm=True),
                                    nn.Conv1d(in_channels=num_channels,
                                              out_channels=num_channels,
                                              kernel_size=2,
                                              groups=num_channels),
                                    nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                    DecoderHead(target_size=target_size, batch_norm=True)])

    def forward(self, x):
        x = self.model[0](x)  # base
        x = self.model[1](x)  # channel-wise fully connected layer
        x = self.model[2](x)  # optional dropout layer
        x = self.model[3](x)  # head
        return x


class AlexNetJigsaw(nn.Module):
    """
    Implements AlexNet model for self-supervised jigsaw puzzle task.
    """
    def __init__(self, num_tiles=9, num_permutations=1000):
        super(AlexNetJigsaw, self).__init__()

        self.num_tiles = num_tiles

        self.siamese_network = nn.ModuleList([AlexNetBase(batch_norm=True),
                                              nn.Linear(256 * 6 * 6, 2048, bias=False),
                                              nn.BatchNorm1d(2048),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(2048, 512, bias=False),
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


class AlexNetColorization(nn.Module):
    """
    Implements AlexNet model for self-supervised colorization task.
    NOTE: We are using a 1D convolutional layer as the channel-wise fully connected layer.
    """
    def __init__(self, target_size, dropout_rate=0.5):
        super(AlexNetColorization, self).__init__()
        num_channels = 256 * 6 * 6
        self.model = nn.ModuleList([AlexNetBase(batch_norm=True),
                                    nn.Conv1d(in_channels=num_channels,
                                              out_channels=num_channels,
                                              kernel_size=2,
                                              groups=num_channels),
                                    nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                    DecoderHead(target_size=target_size, batch_norm=True)])

    def forward(self, x):
        x = self.model[0](x)  # base
        x = self.model[1](x)  # channel-wise fully connected layer
        x = self.model[2](x)  # optional dropout layer
        x = self.model[3](x)  # head
        return x


def alexnet_pretrained(progress=True):
    model = AlexNetClassic()
    state_dict = load_state_dict_from_url(MODEL_URLS['alexnet'], progress=progress)
    state_dict_ = deepcopy(state_dict)
    # Rename parameters to include the starting 'model.<index>' string so they match with our models
    for param in state_dict:
        state_dict_['%s%s' % ('model.%d.' % (0 if 'features' in param else 1), param)] = state_dict_.pop(param)
    model.load_state_dict(state_dict_)
    return model
