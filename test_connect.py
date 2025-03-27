import os
import torch.distributed as dist
import torch
import datetime
import socket
import time
import subprocess

def test_connection(rank, world_size, master_addr, worker_addr=None):
    print(f"I am rank {rank}, attempting connection")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    
    # Explicitly set network interface for Gloo
    if rank == 0: 
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'  # Explicitly use WiFi interface on worker
        print(f"Worker setting GLOO_SOCKET_IFNAME=wlo1")
    elif rank == 1:
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlp4s0'  # Explicitly use WiFi interface on worker
        print(f"Worker setting GLOO_SOCKET_IFNAME=wlp4s0")
    else:
        # Master can set its own appropriate interface
        pass
    
    # Print current environment variables for debugging
    print("Environment variables:")
    for var in ['MASTER_ADDR', 'MASTER_PORT', 'GLOO_SOCKET_IFNAME']:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    # Try disabling IPv6
    os.environ['GLOO_DISABLE_IPV6'] = '1'
    
    try:
        # Use explicit init method
        init_method = f"tcp://{master_addr}:29500"
        print(f"Using init_method: {init_method}")
        
        # Try opening port to ensure it's working
        if rank == 0:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(('0.0.0.0', 29500))
            test_socket.listen(5)
            print("Master successfully bound to port 29500")
            test_socket.close()
        
        # Initialize with more verbose option
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=180)  # Longer timeout
        )
        
        print(f"Rank {rank}: Connected successfully!")
        
        # Simple test to verify it works
        tensor = torch.ones(1) * rank
        dist.all_reduce(tensor)
        print(f"Rank {rank}: Tensor after all_reduce: {tensor.item()}")
        
        # Clean shutdown
        dist.destroy_process_group()
        return True
    
    except Exception as e:
        print(f"Rank {rank}: Connection failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up as best we can
        try:
            dist.destroy_process_group()
        except:
            pass
        
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-addr', type=str, required=True)
    args = parser.parse_args()
    
    # Print detailed network and firewall info
    print("\n==== NETWORK CONFIGURATION ====")
    print("Network interfaces:")
    subprocess.run(['ip', 'addr'], check=False)
    
    if args.rank == 1:  # Worker
        print("\nTesting connection to master:")
        subprocess.run(['ping', '-c', '3', args.master_addr], check=False)
        
        print("\nTesting port connectivity:")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            result = s.connect_ex((args.master_addr, 29500))
            if result == 0:
                print(f"Successfully connected to {args.master_addr}:29500")
            else:
                print(f"Failed to connect to {args.master_addr}:29500, error: {result}")
            s.close()
        except Exception as e:
            print(f"Error testing connection: {e}")
    
    print("\n==== STARTING CONNECTION TEST ====")
    success = test_connection(args.rank, args.world_size, args.master_addr)
    print("Test " + ("succeeded" if success else "failed"))