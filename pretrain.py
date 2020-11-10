# TODO: https://github.com/pytorch/examples/tree/master/imagenet -> Use this for best performance!
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import CIFAR10, ImageNet
from vision.datasets import TinyImageNet, Cityscapes
from torchvision.models import AlexNet
from torchvision.transforms import transforms

from vision.collators import RotationCollator
from utils.helpers import set_seed, accuracy

# Set random seed for reproducibility
SEED = 42
set_seed(seed=SEED)

GPU_ID = 5
DEVICE = torch.device('cuda:%d' % GPU_ID if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Configure hyperparameters
BATCH_SIZE = 128  # 512
LEARNING_RATE = 1E-3
NUM_EPOCHS = 100
MODEL = 'alexnet'
DATASET = 'cityscapes'  # 'tinyimagenet'
TASK = 'rotation'
MODEL_ID = '%s-%s-%s' % (MODEL, DATASET, TASK)

# Define image transform
# NOTE: Change resize to transforms.Resize((64, 64)) below if original size is << like in CIFAR-10
# NOTE: transforms.Resize((256, 256)) for ImageNet
# NOTE: transforms.Resize((64, 64)) for TinyImageNet
# NOTE: transforms.RandomCrop((1024, 1024)) & transforms.Resize((256, 256)) for Cityscapes # TODO: Is this too much downsampling??
transform = transforms.Compose([transforms.RandomCrop((1024, 1024)),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define collator for self-supervised task
rotation_collator = RotationCollator(num_rotation_classes=4)

# Initialize models
model = AlexNet(num_classes=4)
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
print('Loading Datasets...')
# train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
# train_dataset = ImageNet(root='/proj/vondrick/datasets/ImageNet-ILSVRC2012', split='train', download=None, transform=transform)
# test_dataset = ImageNet(root='/proj/vondrick/datasets/ImageNet-ILSVRC2012', split='val', download=None, transform=transform)
# train_dataset = TinyImageNet(root='/proj/vondrick/datasets/tiny-imagenet-200', split='train', transform=transform, in_memory=True)
# test_dataset = TinyImageNet(root='/proj/vondrick/datasets/tiny-imagenet-200', split='val', transform=transform, in_memory=True)
train_dataset = Cityscapes(root='/proj/vondrick/datasets/Cityscapes', split='train', mode='coarse', transform=transform, target_transform=lambda x: 99, in_memory=True)
# NOTE: Turns out `split=train_extra` has many more images but loading it into memory takes ages!
test_dataset = Cityscapes(root='/proj/vondrick/datasets/Cityscapes', split='val', mode='coarse', transform=transform, target_transform=lambda x: 99, in_memory=True)
# NOTE: We are converting labels to 99 because we will not use them!
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=rotation_collator, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=rotation_collator, shuffle=False, pin_memory=True, num_workers=8)

# Train!
best_test_loss = float('inf')
for i in tqdm(range(NUM_EPOCHS), desc='Iterating over Epochs'):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
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
        with open(os.path.join('saved_models', '%s.txt' % MODEL_ID), 'w') as f:
            f.write('Best Test Loss: %0.4f' % best_test_loss)
            f.write('\n')
            f.write('Best Accuracy: %0.4f' % test_acc)
            f.write('\n')
        torch.save(model.state_dict(), os.path.join('saved_models', '%s.pt' % MODEL_ID))

    print('\n')  # Do this in order to go below the second tqdm line
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
