#!/bin/bash -l
#PBS -l walltime=00:10:00,nodes=1:ppn=24:gpus=2,mem=125gb
#PBS -m abe
#PBS -M bures024@umn.edu

cd ~/Forest
module load python2
source activate installs
module load cuda/9.0
python play_bmsb.py
