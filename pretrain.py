# TODO: https://github.com/pytorch/examples/tree/master/imagenet -> Use this for best performance!
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.datasets import CIFAR10
from vision.datasets import TinyImageNet, Cityscapes, ImageNet
from vision.models import AlexNetRotation, AlexNetCounting
from torchvision.transforms import transforms

from vision.collators import RotationCollator, CountingCollator
from vision.losses import rotation_loss, counting_loss
from utils.helpers import set_seed, classification_accuracy

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


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
parser.add_argument('--fraction_data', default=1.0, type=float, help='Fraction of the data sampled from the dataset. Lower for quicker experimentation')
parser.add_argument('--learning_rate_decay', default=0.98, type=float, help='Gamma in exponential learning rate decay')
parser.add_argument('--save', default='./saved_models', type=str, help='saved models directory')

# Arguments for parallelization
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on GPUs')
parser.add_argument('--master_addr', type=str, default='localhost', help='Address of master; master must be able to accept network traffic on the address and port')
parser.add_argument('--master_port', type=str, default='29500', help='Port that master is listening on')

args = parser.parse_args()


def main_worker(rank, world_size):
    # Configure saved models directory
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    print(f'Your model will be save to {args.save}')

    print('RANK: ', rank)
    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        # Use NCCL for GPU training, process must have exclusive access to GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank,  world_size=world_size)

    # Define image transforms for each dataset
    final_transforms = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if args.dataset == 'cifar10':
        img_size = 64 if not args.img_size else args.img_size
        start_transforms = [transforms.Resize((img_size, img_size))]
    elif args.dataset == 'imagenet':
        img_size = 256 if not args.img_size else args.img_size
        start_transforms = [transforms.Resize((img_size, img_size))]
    elif args.dataset == 'tinyimagenet':
        img_size = 64 if not args.img_size else args.img_size
        start_transforms = [transforms.Resize((img_size, img_size))]
    elif args.dataset == 'cityscapes':
        img_size = 512 if not args.img_size else args.img_size
        start_transforms = [transforms.Resize((img_size, img_size))]

    if args.task == 'counting':
        # NOTE: We are multiplying `img_size` by 2 because in `counting` we downsample by 2
        #       If siamese network accepts (200,200) sized images, original size should be (400,400)
        crop_img_size = img_size * 2
        img_size = int(img_size * (4 / 3.5)) * 2
        start_transforms = [transforms.Resize((img_size, img_size)), transforms.RandomCrop((crop_img_size, crop_img_size))]

    # Combine all transformations into one
    transform = transforms.Compose(start_transforms + final_transforms)

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
            model = AlexNetCounting(num_elements=1000)  # TODO: Does 100 make sense here?

    # TODO: Turns out there is no point in running this for AlexNet, as DataParallel
    #       overhead can dominate the runtime if your model is very small or has many small kernels.
        # Parallel
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
            print("Using PyTorch DDP - could not find Apex")
        model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Load datasets
    print('Loading Datasets...')

    if args.dataset == 'cifar10':
        root = './data' if not args.dataset_root else args.dataset_root
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    elif args.dataset == 'imagenet':
        root = '/proj/vondrick/datasets/ImageNet-ILSVRC2012' if not args.dataset_root else args.dataset_root
        train_dataset = ImageNet(root=root, split='train', download=None, fraction_used=args.fraction_data, transform=transform)
        test_dataset = ImageNet(root=root, split='val', download=None, fraction_used=1.0, transform=transform)
    elif args.dataset == 'tinyimagenet':
        root = '/proj/vondrick/datasets/tiny-imagenet-200' if not args.dataset_root else args.dataset_root
        train_dataset = TinyImageNet(root=root, split='train', transform=transform, in_memory=args.in_memory_dataset)
        test_dataset = TinyImageNet(root=root, split='val', transform=transform, in_memory=args.in_memory_dataset)
    elif args.dataset == 'cityscapes':
        root = '/proj/vondrick/datasets/Cityscapes' if not args.dataset_root else args.dataset_root
        train_dataset = Cityscapes(root=root, split='train_extra', mode='coarse', transform=transform, in_memory=args.in_memory_dataset)
        # NOTE: Turns out `split=train_extra` has more images but loading it into memory takes ages!
        test_dataset = Cityscapes(root=root, split='val', mode='coarse', transform=transform, in_memory=args.in_memory_dataset)
        # NOTE: We are converting labels to 99 because we will not use them!
        # Removed target_transform=lambda x: 99
    else:
        raise ValueError('Dataset=%s is not recognized!' % args.dataset)

    if args.local_rank == -1:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  collate_fn=collator, shuffle=True, pin_memory=True,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
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
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)

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
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc = classification_accuracy(y_pred=y_hat, y_true=y)
            elif args.task == 'counting':
                originals, tiles, others = batch
                originals, tiles, others = originals.to(device), tiles.to(device), others.to(device)
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
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    acc = classification_accuracy(y_pred=y_hat, y_true=y)
                elif args.task == 'counting':
                    originals, tiles, others = batch
                    originals, tiles, others = originals.to(device), tiles.to(device), others.to(device)
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
            model_id = '%s-%s-%s' % (args.model, args.dataset, args.task)
            with open(os.path.join(args.save, '%s.txt' % model_id), 'w') as f:
                f.write('Best Test Loss: %0.4f' % best_test_loss)
                f.write('\n')
                if args.task == 'rotation':
                    f.write('Best Accuracy: %0.4f' % test_acc)
                    f.write('\n')
            torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

        print('\n')  # Do this in order to go below the second tqdm line
        if args.task == 'rotation':
            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
            print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
        else:
            print(f'\tTrain Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}')

        # Apply learning rate decay before the beginning of next epoch
        scheduler.step()

    # Cleanup DDP if applicable
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def main():
    n_gpus = torch.cuda.device_count()
    print('We are using %d GPUs' % n_gpus)
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
    # set_seed(seed=args.seed)  # TODO: Apparently this slows down training? -> mentioned https://github.com/pytorch/examples/blob/master/imagenet/main.py

    # NOTE: Below line can be later commented out, for now we'll need it for debugging purposes
    # torch.autograd.set_detect_anomaly(True)

    # Configure device to work on
    # DEVICE = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
    # print("DEVICE FOUND: %s" % DEVICE)

    # Configure model id
    MODEL_ID = '%s-%s-%s' % (args.model, args.dataset, args.task)
    print('MODEL_ID: %s' % MODEL_ID)

    main()
