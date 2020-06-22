#!/bin/bash -l




#BSUB -P cryoem
#BSUB -J vaegan-pipeline
#BSUB -q slacgpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 72:00
#BSUB -e run.err
#BSUB -o run.out
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=no:mps=no"
#BSUB -B

# set up env
source /etc/profile.d/modules.sh
export MODULEPATH=/usr/share/Modules/modulefiles:/opt/modulefiles:/afs/slac/package/singularity/modulefiles
module purge
module load PrgEnv-gcc/4.8.5

module load ray

# change working directory
cd ~/gpfs_home/code/vaetree/

# Set up ray


worker_num=3 # Must be one less that the total number of nodes


nodes=$LSB_HOSTS # Getting the node names -
nodes_array=( $nodes )
echo $nodes
echo $nodes_array

echo ${nodes[0]}
echo ${nodes_array[0]}

node1=${nodes_array[0]}


ip_prefix=$(lsgrun -m $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by ray_pipeline.py launched by Singularity run
export redis_password

lsgrun -m $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  lsgrun -m $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done



# run the command
singularity run --bind /gpfs,/scratch \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane/data:/data \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane:/home \
                --bind /gpfs/slac/cryo/fs1/u/nmiolane/results:/results \
                --nv ../simgs/pipeline.simg
