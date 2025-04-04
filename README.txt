To run the test


## On master node, run it with or without the specified nic name
for example, if not explicit, util function will choose the first NIC.
By default the lo will be chosen

    ./master.sh
    ./master.sh wlp13s0


## On server node, run it like this, same as master, the lo will be choose default
    ./worker.sh 192.168.1.70 1  
    ./worker.sh 192.168.1.70 1 wlp13s0
    

