import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from math import ceil
from random import Random

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

# ----- Dataset Partitioning Classes -----
class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = list(range(data_len))
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

# ----- Helper Functions -----
def get_test_loader():
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    return torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    device = next(model.parameters()).device  # Get device from model
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move to same device as model
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

# ----- Distributed Training Functions -----
def partition_dataset():
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    world_size = dist.get_world_size()
    bsz = 128 // world_size
    partition_sizes = [1.0 / world_size] * world_size
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    
    train_set = torch.utils.data.DataLoader(
        partition,
        batch_size=bsz,
        shuffle=False
    )
    return train_set, bsz

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

# ----- Training Function -----
def run(rank, size):
    # Set device based on availability
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234 + rank)
    
    train_set, bsz = partition_dataset()
    test_loader = get_test_loader() if rank == 0 else None
    
    model = Net().to(device)  # Move model to device first
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0.0
        
        for data, target in train_set:
            # Explicitly move data to the same device as model
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        
        if rank == 0:
            avg_loss = epoch_loss / len(train_set)
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            
            # Test will now handle device transfer automatically
            test_loss, accuracy = test(model, test_loader)
            print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            #No need to save checkpoint but just in case
            #save_checkpoint(model, optimizer, epoch, f'model_epoch_{epoch}.pth')



# ----- Process Initialization -----
def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def main():
    world_size = 2
    mp.set_start_method("spawn", force=True)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
