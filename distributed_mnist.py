import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
#Network function here
import utils

# ----- Model Definition -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----- Setup and Cleanup Functions -----
#Here by set it to 127.0.0.1 allow local worker and master
def setup(rank, world_size, master_addr='127.0.0.1', backend='gloo', interface=None):
    """Initialize distributed training with connection status"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
    
    # Explicitly set network interface for Gloo,here call the thing
    network_interface = utils.get_network_interface(interface)
    if rank == 0:
        network_interface = wlp13s0#On Master
    else:
        network_interface = enp6s0#On Worker
        
    os.environ['GLOO_SOCKET_IFNAME'] = network_interface  # Master interface
    # Print connection status
    if rank == 0:
        print(f"Using network interface: {network_interface}")
        print(f"\nMaster node ready at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        print(f"Using backend: {backend}")
        print("Waiting for worker nodes to connect...")
    else:
        print(f"Using network interface: {network_interface}")
        print(f"Worker node (rank {rank}) attempting to connect...")
    
    # Initialize process group
    try:
        # Use explicit store for better connection
        store = dist.TCPStore(master_addr, 29500, world_size, rank == 0, timeout=datetime.timedelta(seconds=120).total_seconds())
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
        )
        
        # Confirm connection
        if rank == 0:
            print(f"\nAll workers connected! World size: {world_size}\n")
        else:
            print(f"Worker {rank} successfully connected to master")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        raise


    
def cleanup():
    dist.destroy_process_group()

# ----- Training Function -----    
def run(rank, world_size, master_addr='127.0.0.1', backend='gloo', interface=None):
    """Distributed training function"""
    print(f"Rank {rank}: Initializing connection to master at {master_addr}")
    setup(rank, world_size, master_addr, backend, interface)
    
    # Select device based on rank and available devices
    if backend == 'nccl' and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # If we have multiple GPUs, assign each rank to a different GPU
            device_id = rank % torch.cuda.device_count()
            device = torch.device(f'cuda:{device_id}')
            print(f"Rank {rank} using GPU {device_id}")
        else:
            # With only one GPU, we need to use CPU for one process when testing on one machine
            if rank == 0:
                device = torch.device('cuda:0')
                print(f"Rank {rank} using GPU 0")
            else:
                device = torch.device('cpu')
                print(f"Rank {rank} using CPU (avoiding GPU conflict)")
    else:
        # Fall back to CPU if NCCL not used or CUDA not available
        device = torch.device('cpu')
        print(f"Rank {rank} using CPU")
    
    # Model setup - adjust for CPU or single GPU case
    model = Net().to(device)
    
    if backend == 'nccl' and torch.cuda.is_available() and rank == 0:
        # For GPU case
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device_id] if device.type == 'cuda' else None
        )
    else:
        # For CPU case or other ranks with gloo backend
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Data loading
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler
    )
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Training loop
    for epoch in range(3):  # Reduced to 3 epochs for testing
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Only master reports epoch-level stats
        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch} complete")
            print(f"Average loss: {avg_loss:.4f}")
            print("-" * 50)
    
    # Cleanup with status message
    if rank == 0:
        print("\nTraining complete. Shutting down...")
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='Process rank')
    parser.add_argument('--world-size', type=int, required=True, help='World size (total number of processes)')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master node address')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], 
                        help='Backend for distributed training (gloo or nccl)')
    parser.add_argument('--interface', type=str, default=None, 
                        help='Network interface to use (default: auto-detect)')
    args = parser.parse_args()
    
    # Pass the interface parameter to your run function
    run(args.rank, args.world_size, args.master_addr, args.backend, args.interface)