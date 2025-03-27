#!/bin/bash

# Configuration
MASTER_ADDR=$(hostname -I | awk '{print $1}')  # Auto-get IP
MASTER_PORT=29500
WORLD_SIZE=2  # Total number of nodes (master + workers)
NPROC_PER_NODE=1  # Processes per node (typically 1 per GPU)
BACKEND="gloo"  # Use "nccl" for multi-GPU setups across machines


#python distributed_mnist.py --rank 1 --world-size 2  --master-addr 192.168.1.80 --backend gloo

 
# Print configuration
echo "Starting master node with configuration:"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Processes per node: $NPROC_PER_NODE"
echo "Backend: $BACKEND"
echo ""

# Export environment variables
export MASTER_ADDR
export MASTER_PORT

# Run the training script for rank 0 (master)
echo "Starting master process..."
python distributed_mnist.py --rank 0 --world-size $WORLD_SIZE --master-addr $MASTER_ADDR --backend $BACKEND

echo "Master process completed."