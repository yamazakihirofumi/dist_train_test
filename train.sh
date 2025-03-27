#!/bin/bash
#This is a template of the distributed machine learning training

# Cleanup function
#sudo lsof -i :29500
# kill -9 pid


# Trap Ctrl-C
trap cleanup SIGINT

# Launch training
MASTER_ADDR=127.0.0.1
WORLD_SIZE=2
MASTER_PORT=29500

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
   distributed_mnist.py

cleanup
