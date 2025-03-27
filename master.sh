#!/bin/bash
# master.sh - Run this on the master node

# Configuration
MASTER_ADDR=$(hostname -I | awk '{print $1}')  # Auto-get IP
MASTER_PORT=29500
WORLD_SIZE=2  # Total number of nodes (master + slaves)
NPROC_PER_NODE=1  # Processes per node (typically 1 per GPU)

echo "Master starting at ${MASTER_ADDR}:${MASTER_PORT}"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    pkill -f "python.*distributed_mnist.py"
    sleep 1
    echo "Processes killed"
}

# Trap Ctrl-C
trap cleanup SIGINT

# Launch training
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    distributed_mnist.py

cleanup
