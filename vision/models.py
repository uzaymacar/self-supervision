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
    def __init__(self, batch_normalization=True):
        super(AlexNetBase, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.initialize_weights()

    def initialize_weights(self):
        for layer_no in range(len(self.features)):
            if hasattr(self.features[layer_no], 'weight'):
                torch.nn.init.kaiming_normal_(self.features[layer_no].weight)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=1000):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, num_classes))
        self.initialize_weights()

    def initialize_weights(self):
        for layer_no in range(len(self.classifier)):
            if hasattr(self.classifier[layer_no], 'weight'):
                torch.nn.init.kaiming_normal_(self.classifier[layer_no].weight)

    def forward(self, x):
        x = self.classifier(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, output_size=1000):
        super(RegressionHead, self).__init__()
        self.regressor = nn.Sequential(nn.Dropout(),
                                       nn.Linear(256 * 6 * 6, 4096),
                                       # nn.BatchNorm1d(4096),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(),
                                       nn.Linear(4096, 4096),
                                       # nn.BatchNorm1d(4096),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(4096, output_size))
        self.initialize_weights()

    def initialize_weights(self):
        for layer_no in range(len(self.regressor)):
            if hasattr(self.regressor[layer_no], 'weight'):
                torch.nn.init.kaiming_normal_(self.regressor[layer_no].weight)

    def forward(self, x):
        x = self.regressor(x)
        return x


class AlexNetClassic(nn.Module):
    def __init__(self):
        super(AlexNetClassic, self).__init__()
        self.model = nn.ModuleList([AlexNetBase(), ClassificationHead(num_classes=1000)])

    def forward(self, x):
        x = self.model[0](x)
        x = self.model[1](x)
        return x


class AlexNetRotation(nn.Module):
    def __init__(self, num_rotations=4):
        super(AlexNetRotation, self).__init__()
        # self.model = nn.ModuleList([AlexNetBase(), ClassificationHead(num_classes=num_rotations)])
        self.model = AlexNetClassic()
        self.classifier = nn.Linear(1000, num_rotations)

    def forward(self, x):
        # x = self.model[0](x)  # base
        # x = self.model[1](x)  # head
        x = self.model(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


class AlexNetCounting(nn.Module):
    def __init__(self, num_elements=1000):
        """
        AlexNet Base Model + Regression Head; as implemented in "Representation Learning by
        Learning to Count" Noroozi et al.
        :param (int) num_elements: Number of elements that we want to count
        """
        super(AlexNetCounting, self).__init__()
        self.model = nn.ModuleList([AlexNetBase(), RegressionHead(output_size=num_elements)])

    def forward(self, x):
        x = self.model[0](x)  # base
        x = self.model[1](x)  # head
        # NOTE: ReLU is applied at the end because we want the counting vector to be all positive
        x = F.relu(x)
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
