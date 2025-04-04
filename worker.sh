#!/bin/bash

# Trap function for clean exit
cleanup() {
    echo "Caught interrupt signal, shutting down worker..."
    # Kill any child processes (including Python)
    pkill -P $$
    exit 1
}

# Set up trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM


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
RANK=${2:-1}  # Default to rank 1 if not specified
BACKEND="gloo"  # Use gloo backend

INTERFACE=$3   # Explicitly pass "None" for auto-detection if not specified

# Print configuration
echo "Starting worker node with configuration:"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Worker rank: $RANK"
echo "Backend: $BACKEND"
echo ""

# Export environment variables
export MASTER_ADDR
export MASTER_PORT

# Check if interface is specified and not "None"
if [ -n "$INTERFACE" ] && [ "$INTERFACE" != "None" ]; then
    echo "Using network interface: $INTERFACE"
    exec python distributed_mnist.py --rank $RANK --world-size $WORLD_SIZE \
        --master-addr $MASTER_ADDR --backend $BACKEND --interface $INTERFACE
else
    echo "Using auto-detected network interface"
    exec python distributed_mnist.py --rank $RANK --world-size $WORLD_SIZE \
        --master-addr $MASTER_ADDR --backend $BACKEND
fi

echo "Worker process completed."

