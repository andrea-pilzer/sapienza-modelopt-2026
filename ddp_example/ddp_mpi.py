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


def get_resources():

    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # from slurm
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])

    return rank, local_rank, world_size


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def cleanup():
    dist.destroy_process_group()

# Save checkpoint file draft function
def save_checkpoint(ddp_model, world_size):
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    rank = dist.get_rank()
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

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

    rank, local_rank, world_size = get_resources()

    if local_rank is None:
        local_rank = int(os.environ["SLURM_LOCALID"])
        print('Local rank ', local_rank)

    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Create directories outside the PyTorch program
    if rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filepath = os.path.join(model_dir, model_filename)

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18() #pretrained=False)

    device = torch.device("cuda:{}".format(local_rank))

    torch.cuda.set_device(local_rank)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    if torch.backends.cudnn.version() >= 7603 and argv.channels_last:
        model = model.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
    else:
        model = model.to(device)

    if argv.use_zero:
        print_peak_memory("Max memory allocated after creating local model", local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if argv.use_zero:
        print_peak_memory("Max memory allocated after creating DDP", local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

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

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()

    if argv.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=optim.SGD,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
    else:
        optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        t0 = time.time()
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()

        for data in train_loader:
            if torch.backends.cudnn.version() >= 7603 and argv.channels_last:
                inputs, labels = data[0].to(device, memory_format=torch.channels_last), data[1].to(device, memory_format=torch.channels_last)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if argv.use_zero:
                print_peak_memory("Max memory allocated before optimizer step()", local_rank)
            scaler.step(optimizer)
            if argv.use_zero:
                print_peak_memory("Max memory allocated after optimizer step()", local_rank)

            # Updates the scale for next iteration.
            scaler.update()

            #loss.backward()
            #optimizer.step()

        if rank == 0:
            print("Rank: {}, Epoch: {}, Training ...".format(rank, epoch))
            print("Time {} seconds".format(round(time.time() - t0, 2)))

if __name__ == "__main__":
    main()
