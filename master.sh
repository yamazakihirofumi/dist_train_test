#!/bin/bash

#python distributed_mnist.py --rank 1 --world-size 2  --master-addr 192.168.1.80 --backend gloo

# Trap function for clean exit
cleanup() {
    echo "Caught interrupt signal, shutting down master..."
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM

# Configuration
MASTER_ADDR=$(hostname -I | awk '{print $1}')  # Auto-get IP
MASTER_PORT=29500
RANK=0
WORLD_SIZE=2  # Total number of nodes (master + workers)
BACKEND="gloo"  # Use gloo backend
INTERFACE=$1

# Print configuration
echo "Starting master node with configuration:"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Backend: $BACKEND"
echo ""

# Export environment variables
export MASTER_ADDR
export MASTER_PORT

# Print configuration
echo "Starting node: master=$MASTER_ADDR, rank=$RANK"

# Decect if the interface been changed
if [ -n "$INTERFACE" ] && [ "$INTERFACE" != "None" ]; then
    #If it's changed run it with changed param 
    echo "Using network interface: $INTERFACE"
    exec python distributed_mnist.py --rank $RANK --world-size $WORLD_SIZE \
        --master-addr $MASTER_ADDR --backend $BACKEND --interface $INTERFACE
else
    #If it's not change, run without that argument
    echo "Using auto-detected network interface"
    exec python distributed_mnist.py --rank $RANK --world-size $WORLD_SIZE \
        --master-addr $MASTER_ADDR --backend $BACKEND
fi

 

echo "Master process completed."