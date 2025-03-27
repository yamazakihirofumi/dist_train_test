import os
import torch.distributed as dist
import torch
import datetime
import socket
import time

def test_connection(rank, world_size, master_addr):
    print(f"I am rank {rank}, trying to connect to {master_addr}:29500")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    
    try:
        # Use TCP store for explicit connection
        store = dist.TCPStore(master_addr, 29500, world_size, rank == 0)
        
        # Initialize process group with long timeout
        dist.init_process_group(
            backend='gloo',
            store=store,
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-addr', type=str, required=True)
    args = parser.parse_args()
    
    # Print network info
    hostname = socket.gethostname()
    try:
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        print(f"My hostname: {hostname}")
        print(f"My IP addresses: {ip_addresses}")
    except:
        print("Could not get IP addresses")
    
    success = test_connection(args.rank, args.world_size, args.master_addr)
    print("Test " + ("succeeded" if success else "failed"))