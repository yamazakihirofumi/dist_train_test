To run the test


## On master node, run it with or without the specified nic name
for example, if not explicit, util function will choose the first NIC.
By default the lo will be chosen

    ./master.sh
    ./master.sh wlp13s0
    python distributed_mnist.py --rank 0 --world-size 2 \
        --master-addr 192.168.1.70 --backend gloo --interface wlp13s0

## On worker node, run it like this, same as master, the lo will be choose default
    ./worker.sh 192.168.1.70 1  
    ./worker.sh 192.168.1.70 1 enp6s0
    python distributed_mnist.py --rank 1 --world-size 2 \
        --master-addr 192.168.1.70 --backend gloo --interface enp6s0


