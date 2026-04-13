"""
This is a sample script to illustrate DDP with mpirun.

For example to run it on 2 nodes with 1 GPU per node, you can run it interactively on CINECA Leonardo cluster like this:
1. Open an interactive session
salloc -N 2 --ntasks-per-node 1 --cpus-per-task 8 --gres=gpu:1 --mem=8gb -A your_project -p boost_usr_prod -t 01:00:00

2.Import modules
module load profile/deeplrn cineca-ai

3. Run it
mpirun -np 2 -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) -x MASTER_PORT=11234 -x PATH -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib  python ddp_mpi.py --num_epochs 5
"""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import random
import numpy as np
import time
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

import warnings
warnings.filterwarnings('ignore')


def main():

    num_epochs_default = 1
    batch_size_default = 256 # 256 # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "./saved_models"
    model_filename_default = "resnet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # local rank is deprecated for torchrun and new versions of pytorch
    # parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--channels-last", action="store_true", help="Channels last")
    parser.add_argument("--use-zero", action="store_true", help="Zero Redundancy Optimizer")
    argv = parser.parse_args()

    # local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    num_workers = 1

    # Create directories outside the PyTorch program
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filepath = os.path.join(model_dir, model_filename)

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18() #pretrained=False)

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data/cifar-10", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data/cifar-10", train=False, download=True, transform=transform)


if __name__ == "__main__":
    main()
