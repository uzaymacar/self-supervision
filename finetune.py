import os
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import torchvision.models as models
from torchvision.datasets import SVHN
from torchvision.models import AlexNet
from torchvision.transforms import transforms

from utils.helpers import set_seed, freeze_weights, accuracy

# Set random seed for reproducibility
SEED = 42
set_seed(seed=SEED)

DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Configure hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 3E-5  # NOTE: We use 1E-1 times smaller learning rate for finetuning!
NUM_EPOCHS = 100
NUM_CLASSES = 10
PRETRAIN_TYPE = ['self-supervised', 'supervised', 'random'][0]

# Define image transform
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if PRETRAIN_TYPE == 'self-supervised':
    # Load pretrained weights
    model = AlexNet(num_classes=4)
    state_dict = torch.load(os.path.join('saved_models', 'alexnet-cifar10-pretrained-rotation.pt'), map_location=DEVICE)
    state_dict_ = deepcopy(state_dict)
    # Rename parameters to exclude the starting 'module.' string so they match
    # NOTE: We have to do this because of DataParallel  # TODO: See if there is a way around this?
    for param in state_dict:
        state_dict_[param.replace('module.', '')] = state_dict_.pop(param)
    model.load_state_dict(state_dict_)
    # Freeze weights before finetuning
    freeze_weights(model, model_class='alexnet')
elif PRETRAIN_TYPE == 'supervised':
    model = models.alexnet(pretrained=True)
    # Freeze weights before finetuning
    freeze_weights(model, model_class='alexnet')
elif PRETRAIN_TYPE == 'random':
    # NOTE: Here, we are not freezing any weights for fair comparison!
    model = models.alexnet(pretrained=False)
else:
    raise ValueError('PRETRAIN_TYPE=%s is not recognized!' % PRETRAIN_TYPE)

in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

# TODO: Turns out there is no point in running this for AlexNet, as DataParallel
#       overhead can dominate the runtime if your model is very small or has many small kernels.
"""
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
"""
model.to(DEVICE)

# Setup optimizer and loss function
optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Load datasets
train_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Train!
best_test_loss = float('inf')
for i in tqdm(range(NUM_EPOCHS), desc='Iterating over Epochs'):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for batch in train_loader:
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        y_hat = model(x)

        loss = criterion(y_hat, y)
        acc = accuracy(y_pred=y_hat, y_true=y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)

    model.eval()
    epoch_loss, epoch_acc = 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_hat = model(x)

            loss = criterion(y_hat, y)
            acc = accuracy(y_pred=y_hat, y_true=y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    test_loss, test_acc = epoch_loss / len(test_loader), epoch_acc / len(test_loader)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        with open(os.path.join('saved_models', 'alexnet-svhn-finetuned-rotation.txt'), 'w') as f:
            f.write('Best Test Loss: %0.4f' % best_test_loss)
            f.write('\n')
            f.write('Best Accuracy: %0.4f' % test_acc)
        torch.save(model.state_dict(), os.path.join('saved_models', 'alexnet-svhn-finetuned-rotation.pt'))

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
