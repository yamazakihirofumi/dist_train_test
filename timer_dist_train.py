import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
import time
import logging
import socket
import threading
import json
import netifaces

# ----- Logging Setup -----
def setup_logging(rank, log_file="train.log"):
    """Configure logging based on rank"""
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if rank == 0:  # Master node
        # File handler for master
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

# ----- Network Metrics -----
class NetworkMetrics:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.metrics = {
            "job_start_time": None,
            "weight_transmissions": []
        }
        
    def record_job_start(self):
        """Record job start time"""
        self.metrics["job_start_time"] = time.time()
        return self.metrics["job_start_time"]
    
    def start_weight_transmission(self, size_bytes):
        """Record start of weight transmission"""
        transmission = {
            "start_time": time.time(),
            "end_time": None,
            "size_bytes": size_bytes,
            "completed": False
        }
        self.metrics["weight_transmissions"].append(transmission)
        return len(self.metrics["weight_transmissions"]) - 1  # Return index of this transmission
    
    def end_weight_transmission(self, index):
        """Record end of weight transmission"""
        if index < len(self.metrics["weight_transmissions"]):
            self.metrics["weight_transmissions"][index]["end_time"] = time.time()
            self.metrics["weight_transmissions"][index]["completed"] = True
            
            # Calculate metrics
            start = self.metrics["weight_transmissions"][index]["start_time"]
            end = self.metrics["weight_transmissions"][index]["end_time"]
            size = self.metrics["weight_transmissions"][index]["size_bytes"]
            duration = end - start
            throughput = size / duration if duration > 0 else 0  # bytes per second
            
            self.metrics["weight_transmissions"][index]["duration"] = duration
            self.metrics["weight_transmissions"][index]["throughput_bps"] = throughput
            
            return {
                "duration": duration,
                "size_bytes": size,
                "throughput_bps": throughput
            }
        return None
    
    def save_metrics_to_file(self, filename="network_metrics.json"):
        """Save metrics to a JSON file"""
        if self.rank == 0:  # Only master saves metrics
            with open(filename, 'w') as f:
                json.dump(self.metrics, f, indent=4)

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

# ----- Worker Listening Thread -----
class WorkerListenerThread(threading.Thread):
    def __init__(self, rank, world_size, master_addr, backend, logger):
        threading.Thread.__init__(self)
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.backend = backend
        self.logger = logger
        self.ready = False
        self.daemon = True
        
    def run(self):
        """Worker thread that waits for master to be available"""
        if self.rank == 0:  # Master doesn't need to listen
            return
            
        self.logger.info(f"Worker {self.rank} starting listener thread...")
        retry_count = 0
        
        while not self.ready and retry_count < 30:  # Retry for ~5 minutes
            try:
                # Try to connect to master
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                port = int(os.environ.get('MASTER_PORT', '29500'))
                result = s.connect_ex((self.master_addr, port))
                s.close()
                
                if result == 0:  # Connection successful
                    self.logger.info(f"Worker {self.rank} detected master node is available")
                    self.ready = True
                    break
                else:
                    retry_count += 1
                    self.logger.info(f"Worker {self.rank} waiting for master... (attempt {retry_count})")
                    time.sleep(10)  # Wait 10 seconds before retry
            except Exception as e:
                retry_count += 1
                self.logger.info(f"Worker {self.rank} connection error: {e}")
                time.sleep(10)
                
        if not self.ready:
            self.logger.error(f"Worker {self.rank} failed to connect to master after multiple attempts")

# ----- Network Interface Detection -----
def get_interface_for_ip():
    """
    Find a suitable network interface for distributed training.
    Returns the name of a non-loopback interface with a valid IPv4 address.
    """
    # Find first non-loopback interface with a valid IPv4 address
    for interface in netifaces.interfaces():
        # Skip loopback interfaces
        if interface.startswith('lo'):
            continue
            
        addrs = netifaces.ifaddresses(interface)
        # Check for IPv4 addresses
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                if addr['addr'] != '127.0.0.1':
                    return interface
    
    # Fallback to common interface names
    for common_interface in ('eth0', 'en0', 'wlan0', 'ens3'):
        if common_interface in netifaces.interfaces():
            return common_interface
            
    return None

# ----- Setup and Cleanup Functions -----
def setup(rank, world_size, master_addr='127.0.0.1', backend='gloo', logger=None):
    """Initialize distributed training with connection status"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
    
    # Auto-detect network interface for Gloo
    if backend == 'gloo':
        interface_name = get_interface_for_ip()
        if interface_name:
            os.environ['GLOO_SOCKET_IFNAME'] = interface_name
            logger.info(f"Rank {rank} using network interface: {interface_name}")
    
    # Print connection status
    if rank == 0:
        logger.info(f"Master node at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    else:
        logger.info(f"Worker {rank} connecting to {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    
    # Initialize process group
    try:
        store = dist.TCPStore(
            master_addr, 
            29500, 
            world_size, 
            rank == 0, 
            timeout=datetime.timedelta(seconds=60).total_seconds()
        )
        
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
        )
        
        logger.info(f"Rank {rank} connected successfully")
        return True
    except Exception as e:
        logger.error(f"Rank {rank} connection error: {e}")
        return False

def cleanup():
    dist.destroy_process_group()

# ----- Calculate model size -----
def get_model_size_bytes(model):
    """Calculate the size of model parameters in bytes"""
    size_bytes = 0
    for param in model.parameters():
        size_bytes += param.nelement() * param.element_size()
    return size_bytes

# ----- Training Function -----
def run(rank, world_size, master_addr='127.0.0.1', backend='gloo', is_worker_first=False):
    """Distributed training function with network performance tracking"""
    # Setup logging
    logger = setup_logging(rank)
    network_metrics = NetworkMetrics(rank, world_size)
    
    logger.info(f"Rank {rank}: Starting with master at {master_addr}...")
    
    # If this is a worker and worker_first mode is enabled, wait for master
    if rank != 0 and is_worker_first:
        listener = WorkerListenerThread(rank, world_size, master_addr, backend, logger)
        listener.start()
        
        # Wait for the master to be available
        while not listener.ready and listener.is_alive():
            time.sleep(1)
            
        if not listener.ready:
            logger.error(f"Worker {rank} giving up on connecting to master")
            return
    
    # Record job start time (only on master)
    if rank == 0:
        start_time = network_metrics.record_job_start()
        logger.info(f"Job started at {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize distributed backend
    if not setup(rank, world_size, master_addr, backend, logger):
        logger.error(f"Rank {rank} failed to initialize. Exiting.")
        return
    
    # Select device based on rank and available devices
    if backend == 'nccl' and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # If we have multiple GPUs, assign each rank to a different GPU
            device_id = rank % torch.cuda.device_count()
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"Rank {rank} using GPU {device_id}")
        else:
            # With only one GPU, we need to use CPU for one process when testing on one machine
            if rank == 0:
                device = torch.device('cuda:0')
                logger.info(f"Rank {rank} using GPU 0")
            else:
                device = torch.device('cpu')
                logger.info(f"Rank {rank} using CPU (avoiding GPU conflict)")
    else:
        # Fall back to CPU if NCCL not used or CUDA not available
        device = torch.device('cpu')
        logger.info(f"Rank {rank} using CPU")
    
    # Model setup
    model = Net().to(device)
    
    # Calculate model size for network metrics
    model_size_bytes = get_model_size_bytes(model)
    if rank == 0:
        logger.info(f"Model size: {model_size_bytes/1024/1024:.2f} MB")
    
    # Wrap model with DistributedDataParallel
    if backend == 'nccl' and torch.cuda.is_available() and device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device.index] if device.type == 'cuda' else None
        )
    else:
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
            # Record weight transmission start (only on master)
            if rank == 0:
                transmission_idx = network_metrics.start_weight_transmission(model_size_bytes)
                logger.info(f"Starting weight transmission at {time.time()}, size: {model_size_bytes/1024:.2f} KB")
            
            # Synchronize processes to ensure timing accuracy
            dist.barrier()
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            # Another barrier after parameter updates
            dist.barrier()
            
            # Record weight transmission end (only on master)
            if rank == 0:
                metrics = network_metrics.end_weight_transmission(transmission_idx)
                logger.info(f"Completed weight transmission in {metrics['duration']:.4f}s, "
                           f"throughput: {metrics['throughput_bps']/1024/1024:.2f} MB/s")
            
            if rank == 0 and batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Only master reports epoch-level stats
        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch} complete")
            logger.info(f"Average loss: {avg_loss:.4f}")
    
    # Save network metrics to file (only on master)
    if rank == 0:
        network_metrics.save_metrics_to_file()
        logger.info("Training complete. Saved network metrics.")
    
    # Cleanup
    cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='Process rank')
    parser.add_argument('--world-size', type=int, required=True, help='World size (total number of processes)')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master node address')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], 
                        help='Backend for distributed training (gloo or nccl)')
    parser.add_argument('--worker-first', action='store_true', 
                        help='Worker initialization before master (workers will wait for master)')
    parser.add_argument('--interface', type=str, default=None,
                        help='Manually specify network interface name')
    args = parser.parse_args()
    
    # If interface was manually specified, set it now
    if args.interface:
        os.environ['GLOO_SOCKET_IFNAME'] = args.interface
        print(f"Using network interface: {args.interface}")
    
    run(args.rank, args.world_size, args.master_addr, args.backend, args.worker_first)