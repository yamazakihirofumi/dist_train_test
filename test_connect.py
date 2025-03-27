import os
import torch.distributed as dist
import torch
import datetime
import socket
import time

def test_connection(rank, world_size, master_addr, worker_addr=None):
    # For worker, use its own address for receiving connections
    my_addr = worker_addr if rank == 1 and worker_addr else None
    
    print(f"I am rank {rank}, trying to connect to {master_addr}:29500")
    if my_addr:
        print(f"Worker using address {my_addr} for receiving connections")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    
    # For Gloo to use the correct network interface
    if my_addr:
        os.environ['GLOO_SOCKET_IFNAME'] = get_interface_for_ip(my_addr)
    
    try:
        # Initialize with explicit init_method using TCP
        init_method = f"tcp://{master_addr}:29500"
        
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120)
        )
        
        print(f"Rank {rank}: Connected successfully!")
        
        # Simple test to verify it works
        tensor = torch.ones(1) * rank
        
        # Wait to make sure everyone's initialized
        time.sleep(2)
        
        # Perform a collective operation
        dist.all_reduce(tensor)
        print(f"Rank {rank}: Tensor after all_reduce: {tensor.item()}")
        
        dist.destroy_process_group()
        return True
    except Exception as e:
        print(f"Rank {rank}: Connection failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_interface_for_ip(ip):
    """Try to find the network interface name for a given IP address"""
    import subprocess
    try:
        result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        current_iface = None
        for line in lines:
            if line[0] != ' ' and ':' in line:
                current_iface = line.split(':')[1].strip()
            elif f"inet {ip}/" in line and current_iface:
                return current_iface
        return None
    except:
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-addr', type=str, required=True)
    parser.add_argument('--worker-addr', type=str, help='Worker address for receiving connections')
    args = parser.parse_args()
    
    # Print network info
    hostname = socket.gethostname()
    try:
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        print(f"My hostname: {hostname}")
        print(f"My IP addresses: {ip_addresses}")
        
        # Additional network info
        import subprocess
        print("\nDetailed network interfaces:")
        subprocess.run(['ip', 'addr'], check=False)
    except:
        print("Could not get IP addresses")
    
    success = test_connection(args.rank, args.world_size, args.master_addr, args.worker_addr)
    print("Test " + ("succeeded" if success else "failed"))