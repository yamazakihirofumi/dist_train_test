#!/bin/bash
# This scripe run training after the slave node been runned

# Configuration (must match master.sh)
MASTER_ADDR="192.168.1.80"  
MASTER_PORT=29500
WORLD_SIZE=2  # Must match master
NPROC_PER_NODE=1
NODE_RANK=1   # Increment for each slave (1, 2, etc.)

echo "Slave ${NODE_RANK} connecting to ${MASTER_ADDR}:${MASTER_PORT}"

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
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    distributed_mnist.py

cleanup
