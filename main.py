import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import CIFAR10
from torchvision.models import AlexNet
from torchvision.transforms import transforms

from vision.collators import RotationCollator
from utils import set_seed, accuracy

# Set random seed for reproducibility
SEED = 42
set_seed(seed=SEED)

DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Configure hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 3E-4
NUM_EPOCHS = 100

# Define image transform
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define collator for self-supervised task
rotation_collator = RotationCollator(num_rotation_classes=4)

# Initialize models
model = AlexNet(num_classes=4)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
model.to(DEVICE)

# Setup optimizer and loss function
optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Load datasets
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=rotation_collator, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=rotation_collator, shuffle=False)

# Train!
best_test_loss = float('inf')
for i in tqdm(range(NUM_EPOCHS), desc='Iterating over Epochs'):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for batch in tqdm(train_loader, desc='Training Batch'):
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
        torch.save(model.state_dict(), os.path.join('saved_models', 'alexnet-pretrained-rotation.pt'))

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
