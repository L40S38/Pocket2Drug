#!/bin/bash

VAL_FOLD=1
GPU='4'

ulimit -n 10000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python train.py -val_fold ${VAL_FOLD} -gpu ${GPU} > `train_fold${VAL_FOLD}.log` 
