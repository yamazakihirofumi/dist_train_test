import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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
def setup(rank, world_size, master_addr):
    """Setup following the PyTorch tutorial approach"""
    # Set environment variables explicitly
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
    
    print(f"Initializing process group for rank {rank} with master at {master_addr}:29500")
    
    # For Windows backend, use 'gloo' always
    backend = 'gloo'
    
    # Initialize process group exactly as in the tutorial
    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        print(f"Process group initialized for rank {rank}")
        
    except Exception as e:
        print(f"Error initializing process group: {e}")
        raise

def cleanup():
    """Clean up the distributed process group"""
    dist.destroy_process_group()

# ----- Training Function -----
def train(model, device, train_loader, optimizer, epoch, rank):
    """Training function to be used by both master and worker nodes"""
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
        
        if batch_idx % 20 == 0:
            print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Rank {rank} | Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")
    return avg_loss

# ----- Main Function -----
def run(rank, world_size, master_addr):
    """Main distributed training function"""
    # Initialize process group
    setup(rank, world_size, master_addr)
    
    # Always use CPU for simplicity during debugging
    device = torch.device('cpu')
    print(f"Rank {rank} using device: {device}")
    
    # Create model and wrap with DDP
    model = Net().to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    print(f"Rank {rank}: Model wrapped with DDP")
    
    # Create dataset and dataloader
    # Note: each process will download MNIST dataset
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    # Create a distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler
    )
    
    print(f"Rank {rank}: Dataloader ready")
    
    # Optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.5)
    
    # Just train for 1 epoch to test
    print(f"Rank {rank}: Starting training")
    
    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        train(ddp_model, device, train_loader, optimizer, epoch, rank)
    
    print(f"Rank {rank}: Training complete")
    
    # Cleanup
    cleanup()
    print(f"Rank {rank}: Cleanup complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch Distributed MNIST')
    parser.add_argument('--rank', type=int, required=True,
                        help='Node rank')
    parser.add_argument('--world-size', type=int, required=True,
                        help='Total number of processes')
    parser.add_argument('--master-addr', type=str, required=True,
                        help='Master node IP address')
    
    args = parser.parse_args()
    
    # Run the distributed training
    run(args.rank, args.world_size, args.master_addr)