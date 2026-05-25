#!/bin/bash

export HOME=/home/jys5609
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

conda activate cem
cd /mnt/nfs/CEM-RoPose

$HOME/anaconda3/envs/cem/bin/python3 -m Simulation.Cem.dream_cem --num-workers 60 --input-root /mnt/nfs/DreamDataset/Azure --pid=$1
# $HOME/anaconda3/envs/cem/bin/python3 -m Simulation.Cem.dream_cem --num-workers 60 --input-root /mnt/nfs/DreamDataset/Kinect --pid=$1
# $HOME/anaconda3/envs/cem/bin/python3 -m Simulation.Cem.dream_cem --num-workers 60 --input-root /mnt/nfs/DreamDataset/Realsense --pid=$1