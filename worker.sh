#!/bin/bash

# Check if master address is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <master_address> [rank]"
    echo "Example: $0 192.168.1.100 1"
    exit 1
fi

# Configuration
MASTER_ADDR=$1  # Get from command line argument
MASTER_PORT=29500
WORLD_SIZE=2  # Total number of nodes (master + workers)
NPROC_PER_NODE=1  # Processes per node (typically 1 per GPU)
RANK=${2:-1}  # Default to rank 1 if not specified
BACKEND="gloo"  # Use "nccl" for multi-GPU setups across machines

# Print configuration
echo "Starting worker node with configuration:"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Worker rank: $RANK"
echo "Processes per node: $NPROC_PER_NODE"
echo "Backend: $BACKEND"
echo ""

# Export environment variables
export MASTER_ADDR
export MASTER_PORT

# Run the training script for the worker
echo "Starting worker process..."
python distributed_mnist.py --rank $RANK --world-size $WORLD_SIZE --master-addr $MASTER_ADDR --backend $BACKEND

echo "Worker process completed."