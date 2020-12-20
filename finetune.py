import argparse
import os
import json
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import torchvision.models as models
from torchvision.datasets import SVHN, CIFAR10, ImageFolder
from torchvision.transforms import transforms

from vision.models import AlexNetRotation, AlexNetCounting, AlexNetJigsaw, AlexNetColorization, AlexNetContext, alexnet_pretrained
from vision.losses import classification_loss, counting_loss
from utils.helpers import set_seed, freeze_weights, classification_accuracy


parser = argparse.ArgumentParser(description='Script for finetuning on self-supervised tasks.')

parser.add_argument('--img_size', default=None, type=int, help='Pass the img_size for resizing')
parser.add_argument('--batch_size', default=512, type=int, help='Mini-batch size for training')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate for training')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay term for L2-regularization')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs for training')
parser.add_argument('--dataset', default='svhn', choices=['svhn', 'cifar10', 'flower'],
                    help='(Large) vision dataset')
parser.add_argument('--dataset_root', default=None, type=str, help='Path to the root directory of the chosen dataset')
parser.add_argument('--pretrain_type', default='self-supervised', choices=['self-supervised', 'supervised', 'random'],
                    help='Pretrain task type')
parser.add_argument('--model_path', default=None, type=str,
                    help='Path to the pretrain model (ONLY used if pretrain task type is self-supervised)')
parser.add_argument('--task', default='rotation', choices=['rotation', 'counting', 'context', 'colorization', 'jigsaw'],
                    help='Self-supervised task (ONLY used if pretrain task type is self-supervised)')
parser.add_argument('--freeze_layer', default=3, type=int, choices=[0, 1, 2, 3, 4, 5],
                    help='Freeze conv layer')
parser.add_argument('--seed', default=42, type=int, help='Set seeds for reproducibility')
parser.add_argument('--in_memory_dataset', default=False, action='store_true',
                    help='Indicate to load dataset to memory (if applicable)')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for the data loaders')
parser.add_argument('--learning_rate_decay', default=0.98, type=float, help='Gamma in exponential learning rate decay')
parser.add_argument('--continue_from_checkpoint', default=False, action='store_true',
                    help='Indicate the load model from checkpoint and continue training')
parser.add_argument('--save', default='./saved_models', type=str, help='Saved models directory')
parser.add_argument('--distributed', default=False, action='store_true',
                    help='Whether the pretrained model is distributed training')

args = parser.parse_args()


def main():
    # Configure model ID
    if args.pretrain_type == 'self-supervised':
        model_id = f'{args.pretrain_type}-{args.task}-{args.dataset}-finetuned'
    else:
        model_id = f'{args.pretrain_type}-{args.dataset}-finetuned'
    print(f'MODEL_ID: {model_id}')

    # Configure saved models directory
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    print(f'Your model will be saved to {args.save}')

    # Set random seed for reproducibility
    set_seed(seed=args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("DEVICE FOUND: %s" % device)

    # Define image transform
    if args.dataset == 'svhn' or args.dataset == 'cifar10':
        num_classes = 10
        img_size = 64 if not args.img_size else args.img_size
    elif args.dataset == 'flower':
        num_classes = 102
        img_size = 224 if not args.img_size else args.img_size
    else:
        raise ValueError('dataset=%s is not recognized!' % args.dataset)
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # Load datasets
    if args.dataset == 'svhn':
        train_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'flower':
        flower_dir = os.path.join('.', 'data', 'flower_data')
        train_dataset = ImageFolder(root=os.path.join(flower_dir, 'train'), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(flower_dir, 'valid'), transform=transform)
    else:
        raise ValueError('dataset=%s is not recognized!' % args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Prepare pretrained model
    if args.pretrain_type == 'self-supervised':
        # Load pretrained weights
        if args.task == 'rotation':
            model = AlexNetRotation(num_rotations=4)
        elif args.task == 'counting':
            model = AlexNetCounting(num_elements=1000)
        elif args.task == 'context':
            model = AlexNetContext(target_size=224)
        elif args.task == 'jigsaw':
            model = AlexNetJigsaw(num_tiles=9, num_permutations=1000)
        elif args.task == 'colorization':
            model = AlexNetColorization(target_size=224)
        state_dict = torch.load(args.model_path, map_location=device)
        state_dict_ = deepcopy(state_dict)

        # Rename parameters to exclude the starting 'module.' string so they match
        # NOTE: We have to do this because of DataParallel  # TODO: See if there is a way around this?
        if args.distributed:
            for param in state_dict:
                state_dict_[param.replace('module.', '')] = state_dict_.pop(param)
        model.load_state_dict(state_dict_)
        # Freeze weights before finetuning
        freeze_weights(model, model_class=f'alexnet-{args.task}', freeze_layer=args.freeze_layer)
    elif args.pretrain_type == 'supervised':
        model = models.alexnet(pretrained=True)
        # Freeze weights before finetuning
        freeze_weights(model, model_class='alexnet', freeze_layer=args.freeze_layer)
    elif args.pretrain_type == 'random':
        # NOTE: Here, we are STILL freezing weights for fair comparison
        model = models.alexnet(pretrained=False)
        freeze_weights(model, model_class='alexnet', freeze_layer=args.freeze_layer)
    else:
        raise ValueError('PRETRAIN_TYPE=%s is not recognized!' % args.pretrain_type)

    # Transfer learning here
    if args.pretrain_type == 'self-supervised':
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    # TODO: Turns out there is no point in running this for AlexNet, as DataParallel
    #       overhead can dominate the runtime if your model is very small or has many small kernels.
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    """
    model.to(device)

    # Setup optimizer and loss function
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)

    # Only doing classification task now
    criterion = classification_loss

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # Training & Evaluation
    best_test_loss = float('inf')
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        # -------------------------------- TRAINING ------------------------------------
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            acc = classification_accuracy(y_pred=y_hat, y_true=y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)

        # -------------------------------- EVALUATION ------------------------------------
        model.eval()
        epoch_loss, epoch_acc = 0.0, 0.0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc = classification_accuracy(y_pred=y_hat, y_true=y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        test_loss, test_acc = epoch_loss / len(test_loader), epoch_acc / len(test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            history['best_test_loss'] = best_test_loss
            history['best_acc'] = test_acc
            torch.save(model.state_dict(), os.path.join(args.save, f'{model_id}.pt'))

        print('\n')  # Do this in order to go below the second tqdm line
        print(f'\tTrain Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2%}')
        print(f'\tTest Loss:  {test_loss:.4f} | Test Accuracy:  {test_acc:.2%}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Apply learning rate decay before the beginning of next epoch
        scheduler.step()

    # Save the history
    history_path = os.path.join(args.save, f'{model_id}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f'\nHistory is saved to {history_path}')


if __name__ == '__main__':
    main()