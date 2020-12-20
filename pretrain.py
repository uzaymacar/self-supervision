# TODO: https://github.com/pytorch/examples/tree/master/imagenet -> Use this for best performance!
import argparse
import os
from copy import deepcopy
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau

from torchvision.datasets import CIFAR10
from vision.datasets import TinyImageNet, Cityscapes, ImageNet
from vision.models import AlexNetClassic, AlexNetRotation, AlexNetCounting, AlexNetContext, AlexNetJigsaw, AlexNetColorization
from torchvision.transforms import transforms

from vision.collators import RotationCollator, CountingCollator, ContextCollator, JigsawCollator, ColorizationCollator
from vision.losses import classification_loss, counting_loss, reconstruction_loss
from utils.helpers import set_seed, classification_accuracy

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description='Script for pretraining on self-supervised tasks.')
parser.add_argument('--img_size', default=None, type=int, help='Pass the img_size for resizing')
parser.add_argument('--batch_size', default=512, type=int, help='Mini-batch size for training')
# parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='Optimizer to use in training')
# parser.add_argument('--learning_rate', default=3e-4, type=float, help='Learning rate for training')
# parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay term for L2-regularization')
# parser.add_argument('--learning_rate_decay', default=0.98, type=float, help='Gamma in exponential learning rate decay')
parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs for training')
parser.add_argument('--model', default='alexnet', choices=['alexnet'], help='Base model to pretrain')
parser.add_argument('--dataset', default='tinyimagenet', choices=['cityscapes', 'tinyimagenet', 'imagenet', 'cifar10'], help='(Large) vision dataset')
parser.add_argument('--dataset_root', default=None, type=str, help='Path to the root directory of the chosen dataset')
parser.add_argument('--task', default='rotation',
                    choices=['classification', 'rotation', 'counting', 'jigsaw', 'context', 'colorization'],
                    help='Self-supervised task')
parser.add_argument('--seed', default=42, type=int, help='Set seeds for reproducibility')
parser.add_argument('--in_memory_dataset', default=False, action='store_true', help='Indicate to load dataset to memory (if applicable)')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for the data loaders')
parser.add_argument('--fraction_data', default=1.0, type=float, help='Fraction of the data sampled from the dataset. Lower for quicker experimentation')
parser.add_argument('--continue_from_checkpoint', default=False, action='store_true', help='Indicate the load model from checkpoint and continue training')
parser.add_argument('--save', default='./saved_models', type=str, help='saved models directory')

# Arguments for parallelization
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on GPUs')
parser.add_argument('--master_addr', type=str, default='localhost', help='Address of master; master must be able to accept network traffic on the address and port')
parser.add_argument('--master_port', type=str, default='29500', help='Port that master is listening on')

args = parser.parse_args()


def main_worker(rank, world_size):
    # NOTE: Below line can be later commented out, for now we'll need it for debugging purposes
    # torch.autograd.set_detect_anomaly(True)

    # Configure model ID
    model_id = '%s-%s-%s-%s' % (args.model, args.dataset, str(args.fraction_data), args.task)
    print('MODEL_ID: %s' % model_id)

    # Configure saved models directory
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    print(f'Your model will be save to {args.save}')

    # Create the logging file
    if args.local_rank != -1 and rank == 0 or args.local_rank == -1:
        with open(os.path.join(args.save, '%s.csv' % model_id), 'w') as f:
            writer = csv.writer(f)
            if args.task in ['classification', 'rotation', 'jigsaw']:
                writer.writerow(['train_loss', 'train_acc', 'test_loss', 'test_acc', 'learning_rate'])
            elif args.task in ['counting', 'context', 'colorization']:
                writer.writerow(['train_loss', 'test_loss', 'learning_rate'])

    print('RANK: ', rank)
    if args.local_rank == -1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        # Use NCCL for GPU training, process must have exclusive access to GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank,  world_size=world_size)
        process_group = torch.distributed.new_group(list(range(world_size)))

    # Define default image sizes for each dataset
    if args.dataset == 'cifar10':
        img_size = 64 if not args.img_size else args.img_size
    elif args.dataset == 'imagenet':
        img_size = 224 if not args.img_size else args.img_size
    elif args.dataset == 'tinyimagenet':
        img_size = 64 if not args.img_size else args.img_size
    elif args.dataset == 'cityscapes':
        img_size = 512 if not args.img_size else args.img_size
    else:
        raise ValueError('dataset=%s is not recognized!' % args.dataset)

    # Define image transforms for each task
    end_transforms = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    if args.task == 'colorization':
        end_transforms.insert(0, transforms.ToPILImage(mode='LAB'))

    # NOTE: We are multiplying `img_size` by 2 because in `counting` we downsample by 2
    #       If siamese network accepts (224,224) sized images, original size should be (448,448)
    if args.task == 'counting':
        img_size *= 2

    crop_img_size = img_size
    img_size = int(img_size * (4 / 3.5))
    start_transforms = [transforms.Resize(img_size), transforms.RandomCrop(crop_img_size)]

    # Combine all transformations into one, considering different split-augmentation strategies
    train_transform = transforms.Compose(start_transforms + [transforms.RandomHorizontalFlip()] + end_transforms)
    test_transform = transforms.Compose([start_transforms[0], transforms.CenterCrop(crop_img_size)] + end_transforms)

    if args.local_rank != -1 and rank == 0 or args.local_rank == -1:
        print('Train Transform: ', train_transform)
        print('Test Transform: ', test_transform)

    # Define collator for each self-supervised task
    if args.task == 'classification':
        collator = None
    elif args.task == 'rotation':
        collator = RotationCollator(num_rotations=4, rotation_procedure='all')
        if collator.rotation_procedure == 'all':
            print('Running rotation with `rotation_procedure=all`: effective batch size is %d' %
                  int(args.batch_size * collator.num_rotations))
    elif args.task == 'counting':
        collator = CountingCollator(num_tiles=4, grayscale_probability=0.6)
    elif args.task == 'context':
        collator = ContextCollator(mask_method='central_block', fill_value=(0.485, 0.456, 0.406))
    elif args.task == 'jigsaw':
        collator = JigsawCollator(num_tiles=9, num_permutations=1000, grayscale_probability=0.3)
    elif args.task == 'colorization':
        collator = ColorizationCollator()

    # Initialize model
    if args.model == 'alexnet':
        if args.task == 'classification':
            model = AlexNetClassic()
        elif args.task == 'rotation':
            model = AlexNetRotation(num_rotations=4)
        elif args.task == 'counting':
            model = AlexNetCounting(num_elements=1000)
        elif args.task == 'context':
            model = AlexNetContext(target_size=crop_img_size)
        elif args.task == 'jigsaw':
            model = AlexNetJigsaw(num_tiles=9, num_permutations=1000)
        elif args.task == 'colorization':
            model = AlexNetColorization(target_size=crop_img_size)
        else:
            raise ValueError('task=%s is not recognized!' % args.task)
    else:
        raise ValueError('model=%s is not recognized!' % args.model)

    # Print architecture once
    if args.local_rank != -1 and rank == 0 or args.local_rank == -1:
        print(model)

    # Load saved model if applicable
    if args.continue_from_checkpoint:
        load_path = os.path.join(args.save, '%s.pt' % model_id)
        print('Loading learned weights from %s' % load_path)
        state_dict = torch.load(load_path, map_location=device)
        state_dict_ = deepcopy(state_dict)
        # Rename parameters to exclude the starting 'module.' string so they match
        # NOTE: We have to do this because of DataParallel saving parameters starting with 'module.'
        for param in state_dict:
            state_dict_[param.replace('module.', '')] = state_dict_.pop(param)
        model.load_state_dict(state_dict_)
    else:
        print('Initializing model from scratch')

    if args.local_rank != -1 and args.task == 'counting':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    if torch.cuda.device_count() > 1 and args.local_rank == -1:
        model = nn.DataParallel(model)
        model.to(device)
    elif args.local_rank == -1:
        model.to(device)
    else:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            from torch.nn.parallel import DistributedDataParallel as DDP
            print('Using PyTorch DDP - could not find Apex')
        model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
                    # broadcast_buffers=False if args.task == 'counting' else True)
        # TODO: Only set `broadcast_buffers` to False for the `counting` task, for now...
        # NOTE: `broadcast_buffers` set to False for https://github.com/pytorch/pytorch/issues/22095

    # Load datasets
    print('Loading Datasets...')
    if args.dataset == 'cifar10':
        root = './data' if not args.dataset_root else args.dataset_root
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'imagenet':
        root = '/proj/vondrick/datasets/ImageNet-ILSVRC2012' if not args.dataset_root else args.dataset_root
        train_dataset = ImageNet(root=root, split='train', download=None, fraction_used=args.fraction_data, transform=train_transform)
        test_dataset = ImageNet(root=root, split='val', download=None, fraction_used=1.0, transform=test_transform)
        # TODO: Change fraction used back to 1.0 for test dataset!
    elif args.dataset == 'tinyimagenet':
        root = '/proj/vondrick/datasets/tiny-imagenet-200' if not args.dataset_root else args.dataset_root
        train_dataset = TinyImageNet(root=root, split='train', transform=train_transform, in_memory=args.in_memory_dataset)
        test_dataset = TinyImageNet(root=root, split='val', transform=test_transform, in_memory=args.in_memory_dataset)
    elif args.dataset == 'cityscapes':
        root = '/proj/vondrick/datasets/Cityscapes' if not args.dataset_root else args.dataset_root
        train_dataset = Cityscapes(root=root, split='train_extra', mode='coarse', transform=train_transform, in_memory=args.in_memory_dataset)
        # NOTE: Turns out `split=train_extra` has more images but loading it into memory takes ages!
        test_dataset = Cityscapes(root=root, split='val', mode='coarse', transform=test_transform, in_memory=args.in_memory_dataset)
        # NOTE: We are converting labels to 99 because we will not use them!
        # Removed target_transform=lambda x: 99
    else:
        raise ValueError('dataset=%s is not recognized!' % args.dataset)

    if args.local_rank == -1:
        n_gpus = torch.cuda.device_count()
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size // n_gpus,
                                  collate_fn=collator, shuffle=True, pin_memory=True,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size // n_gpus,
                                 collate_fn=collator, shuffle=False, pin_memory=True,
                                 num_workers=args.num_workers)
    else:
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(dataset=test_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  collate_fn=collator, shuffle=False, pin_memory=True,
                                  num_workers=args.num_workers, sampler=train_sampler)
        # NOTE: Train loader's shuffle is made False; using DistributedSampler instead
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 collate_fn=collator, shuffle=False, pin_memory=True,
                                 num_workers=args.num_workers, sampler=test_sampler)

    # Setup optimizer and loss function
    if args.task == 'classification':
        optimizer = Adam(model.parameters(), lr=4e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3)
    elif args.task == 'counting':
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3)
    elif args.task == 'rotation':
        optimizer = Adam(model.parameters(), lr=3e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)
    elif args.task == 'jigsaw':
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3)
    elif args.task == 'context':
        optimizer = Adam(model.parameters(), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)
    elif args.task == 'colorization':
        optimizer = Adam(model.parameters(), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3)

    if args.task in ['classification', 'rotation', 'jigsaw']:
        criterion = classification_loss
    elif args.task == 'counting':
        criterion = counting_loss
    elif args.task in ['context', 'colorization']:
        criterion = reconstruction_loss

    # Training & Evaluation
    best_test_loss = float('inf')
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        # -------------------------------- TRAINING ------------------------------------
        model.train()
        epoch_loss = 0.0
        if args.task in ['classification', 'rotation', 'jigsaw']:
            epoch_acc = 0.0

        for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
            if args.task in ['classification', 'rotation', 'jigsaw']:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc = classification_accuracy(y_pred=y_hat, y_true=y)
            elif args.task == 'counting':
                x, tiles = batch
                x, tiles = x.to(device), tiles.to(device)
                originals_y_hat = model(x)
                tiles_y_hat = torch.stack([model(tile) for tile in tiles]).to(device)
                loss = criterion(originals_y_hat, tiles_y_hat)
            elif args.task == 'context':
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y, mask)
            elif args.task == 'colorization':
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y, mask=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if args.task in ['classification', 'rotation', 'jigsaw']:
                epoch_acc += acc.item()

        train_loss = epoch_loss / len(train_loader)
        if args.task in ['classification', 'rotation', 'jigsaw']:
            train_acc = epoch_acc / len(train_loader)

        # -------------------------------- EVALUATION ------------------------------------
        model.eval()
        epoch_loss = 0.0
        if args.task in ['classification', 'rotation', 'jigsaw']:
            epoch_acc = 0.0

        with torch.no_grad():
            for batch in test_loader:
                if args.task in ['classification', 'rotation', 'jigsaw']:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    acc = classification_accuracy(y_pred=y_hat, y_true=y)
                elif args.task == 'counting':
                    x, tiles = batch
                    x, tiles = x.to(device), tiles.to(device)
                    originals_y_hat = model(x)
                    tiles_y_hat = torch.stack([model(tile) for tile in tiles]).to(device)
                    loss = criterion(originals_y_hat, tiles_y_hat)
                elif args.task == 'context':
                    x, y, mask = batch
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y, mask)
                elif args.task == 'colorization':
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y, mask=None)

                epoch_loss += loss.item()
                if args.task in ['classification', 'rotation', 'jigsaw']:
                    epoch_acc += acc.item()

        test_loss = epoch_loss / len(test_loader)
        if args.task in ['classification', 'rotation', 'jigsaw']:
            test_acc = epoch_acc / len(test_loader)

        # Apply learning rate decay before next epoch, if validation accuracy hasn't changed
        # if test_loss >= best_test_loss:
        scheduler.step(test_loss)

        # -------------------------------- LOGGING ------------------------------------
        # All processes should see same parameters; saving & logging one process is sufficient
        if args.local_rank != -1 and rank == 0 or args.local_rank == -1:
            print('\n')  # Do this in order to go below the second tqdm line
            if args.task in ['classification', 'rotation', 'jigsaw']:
                print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
                print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
            else:
                print(f'\tTrain Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}')

            with open(os.path.join(args.save, '%s.csv' % model_id), 'a') as f:
                writer = csv.writer(f)
                if args.task in ['classification', 'rotation', 'jigsaw']:
                    writer.writerow(['%0.3f' % train_loss, '%0.2f' % (train_acc * 100),
                                     '%0.3f' % test_loss, '%0.2f' % (test_acc * 100),
                                     optimizer.param_groups[0]['lr']])
                else:
                    writer.writerow(['%0.3f' % train_loss,
                                     '%0.3f' % test_loss,
                                     optimizer.param_groups[0]['lr']])

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model_id = '%s-%s-%s' % (args.model, args.dataset, args.task)
                with open(os.path.join(args.save, '%s.txt' % model_id), 'w') as f:
                    f.write('Best Test Loss: %0.4f\n' % best_test_loss)
                    if args.task in ['classification', 'rotation', 'jigsaw']:
                        f.write('Best Accuracy: %0.4f\n' % test_acc)
                torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

    # Cleanup DDP if applicable
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def main():
    # Configure number of GPUs to be used
    n_gpus = torch.cuda.device_count()
    print('We are using %d GPUs' % n_gpus)

    # Spawn the training process
    if args.local_rank == -1:
        main_worker(rank=-1, world_size=1)
    else:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        print('Spawning...')
        # Number of processes spawned is equal to the number of GPUs available
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus,))


if __name__ == '__main__':
    # Set random seed for reproducibility
    # NOTE: Settings seeds requires cuda.deterministic = True, which slows things down considerably
    # set_seed(seed=args.seed)

    main()
