# TODO: https://github.com/pytorch/examples/tree/master/imagenet -> Use this for best performance!
import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision.datasets import CIFAR10, ImageNet
from vision.datasets import TinyImageNet, Cityscapes
from vision.models import AlexNetRotation, AlexNetCounting
from torchvision.transforms import transforms

from vision.collators import RotationCollator, CountingCollator
from vision.losses import rotation_loss, counting_loss
from utils.helpers import set_seed, classification_accuracy


parser = argparse.ArgumentParser(description='Script for pretraining on self-supervised tasks.')
parser.add_argument('--img_size', default=None, type=int, help='Pass the img_size for resizing')
parser.add_argument('--batch_size', default=512, type=int, help='Mini-batch size for training')
parser.add_argument('--learning_rate', default=3e-4, type=float, help='Learning rate for training')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay term for L2-regularization')
parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs for training')
parser.add_argument('--model', default='alexnet', choices=['alexnet'], help='Base model to pretrain')
parser.add_argument('--dataset', default='tinyimagenet', choices=['cityscapes', 'tinyimagenet', 'imagenet', 'cifar10'], help='(Large) vision dataset')
parser.add_argument('--dataset_root', default=None, type=str, help='Path to the root directory of the chosen dataset')
parser.add_argument('--task', default='rotation', choices=['rotation', 'counting'], help='Self-supervised task')
parser.add_argument('--seed', default=42, type=int, help='Set seeds for reproducibility')
parser.add_argument('--in_memory_dataset', default=False, action='store_true', help='Indicate to load dataset to memory (if applicable)')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for the data loaders')
parser.add_argument('--gpu_id', default=7, type=int, help='This is the GPU to use in server [for UZAY]')
args = parser.parse_args()

# Set random seed for reproducibility
set_seed(seed=args.seed)

# NOTE: Below line can be later commented out, for now we'll need it for debugging purposes
torch.autograd.set_detect_anomaly(True)

# Configure device to work on
DEVICE = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Configure model id
MODEL_ID = '%s-%s-%s' % (args.model, args.dataset, args.task)
print('MODEL_ID: %s' % MODEL_ID)

# Define image transforms for each dataset
final_transforms = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
if args.dataset == 'cifar10':
    img_size = 64 if not args.img_size else args.img_size
    transform = transforms.Compose([transforms.Resize((img_size, img_size))] + final_transforms)
elif args.dataset == 'imagenet':
    img_size = 256 if not args.img_size else args.img_size
    transform = transforms.Compose([transforms.Resize((img_size, img_size))] + final_transforms)
elif args.dataset == 'tinyimagenet':
    # TODO: I have to use 288 to be able to run COUNTING with tinyimagenet but this is probably not OK
    img_size = 64 if not args.img_size else args.img_size
    # transform = transforms.Compose([transforms.Resize((img_size, img_size))] + final_transforms)
    transform = transforms.Compose([*final_transforms])
    if args.task == 'counting':
        img_size = 288
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomCrop((256, 256))] + final_transforms)
        # NOTE: The above transform was specified in the paper so this is what we do here as well!

elif args.dataset == 'cityscapes':
    img_size = 256 if not args.img_size else args.img_size
    transform = transforms.Compose([transforms.RandomCrop((1024, 1024)), transforms.Resize((img_size, img_size))] + final_transforms)
    # TODO: Is this too much downsampling?

# Define collator for each self-supervised task
if args.task == 'rotation':
    collator = RotationCollator(num_rotations=4)
elif args.task == 'counting':
    collator = CountingCollator(num_tiles=4)

# Initialize model
if args.model == 'alexnet':
    if args.task == 'rotation':
        model = AlexNetRotation(num_rotations=4)
    elif args.task == 'counting':
        model = AlexNetCounting()

# TODO: Turns out there is no point in running this for AlexNet, as DataParallel
#       overhead can dominate the runtime if your model is very small or has many small kernels.
import torch.nn as nn
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
model.to(DEVICE)

# Load datasets
print('Loading Datasets...')
if args.dataset == 'cifar10':
    root = './data' if not args.dataset_root else args.dataset_root
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
elif args.dataset == 'imagenet':
    root = '/proj/vondrick/datasets/ImageNet-ILSVRC2012' if not args.dataset_root else args.dataset_root
    train_dataset = ImageNet(root=root, split='train', download=None, transform=transform)
    test_dataset = ImageNet(root=root, split='val', download=None, transform=transform)
elif args.dataset == 'tinyimagenet':
    root = '/proj/vondrick/datasets/tiny-imagenet-200' if not args.dataset_root else args.dataset_root
    train_dataset = TinyImageNet(root=root, split='train', transform=transform, in_memory=args.in_memory_dataset)
    test_dataset = TinyImageNet(root=root, split='val', transform=transform, in_memory=args.in_memory_dataset)
elif args.dataset == 'cityscapes':
    root = '/proj/vondrick/datasets/Cityscapes' if not args.dataset_root else args.dataset_root
    train_dataset = Cityscapes(root=root, split='train', mode='coarse', transform=transform, target_transform=lambda x: 99, in_memory=args.in_memory_dataset)
    # NOTE: Turns out `split=train_extra` has more images but loading it into memory takes ages!
    test_dataset = Cityscapes(root=root, split='val', mode='coarse', transform=transform, target_transform=lambda x: 99, in_memory=args.in_memory_dataset)
    # NOTE: We are converting labels to 99 because we will not use them!

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False, pin_memory=True, num_workers=args.num_workers)

# Setup optimizer and loss function
optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# TODO: Add a scheduler

if args.task == 'rotation':
    criterion = rotation_loss
elif args.task == 'counting':
    criterion = counting_loss

# Training & Evaluation
best_test_loss = float('inf')
for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
    # -------------------------------- TRAINING ------------------------------------
    model.train()

    epoch_loss = 0.0
    if args.task == 'rotation':
        epoch_acc = 0.0

    for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
        optimizer.zero_grad()

        if args.task == 'rotation':
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            acc = classification_accuracy(y_pred=y_hat, y_true=y)
        elif args.task == 'counting':
            originals, tiles, others = batch
            originals, tiles, others = originals.to(DEVICE), tiles.to(DEVICE), others.to(DEVICE)
            originals_y_hat, others_y_hat = model(originals), model(others)
            tiles_y_hat = torch.stack([model(tile) for tile in tiles])
            loss = criterion(originals_y_hat, tiles_y_hat, others_y_hat)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if args.task == 'rotation':
            epoch_acc += acc.item()

    train_loss = epoch_loss / len(train_loader)
    if args.task == 'rotation':
        train_acc = epoch_acc / len(train_loader)

    # -------------------------------- EVALUATION ------------------------------------
    model.eval()
    epoch_loss = 0.0
    if args.task == 'rotation':
        epoch_acc = 0.0

    with torch.no_grad():
        for batch in test_loader:
            if args.task == 'rotation':
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc = classification_accuracy(y_pred=y_hat, y_true=y)
            elif args.task == 'counting':
                originals, tiles, others = batch
                originals, tiles, others = originals.to(DEVICE), tiles.to(DEVICE), others.to(DEVICE)
                originals_y_hat, others_y_hat = model(originals), model(others)
                tiles_y_hat = torch.stack([model(tile) for tile in tiles])
                loss = criterion(originals_y_hat, tiles_y_hat, others_y_hat)

            epoch_loss += loss.item()
            if args.task == 'rotation':
                epoch_acc += acc.item()

    test_loss = epoch_loss / len(test_loader)
    if args.task == 'rotation':
        test_acc = epoch_acc / len(test_loader)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        with open(os.path.join('saved_models', '%s.txt' % MODEL_ID), 'w') as f:
            f.write('Best Test Loss: %0.4f' % best_test_loss)
            f.write('\n')
            if args.task == 'rotation':
                f.write('Best Accuracy: %0.4f' % test_acc)
                f.write('\n')
        torch.save(model.state_dict(), os.path.join('saved_models', '%s.pt' % MODEL_ID))

    print('\n')  # Do this in order to go below the second tqdm line
    if args.task == 'rotation':
        print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
        print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
    else:
        print(f'\tTrain Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}')
