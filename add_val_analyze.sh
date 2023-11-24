#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python analyze_data_para2.py -val_pocket_sample_dir ~/add_val_data/val_pockets_sample_2048/ -config_yaml ~/add_val_data/pocket-smiles.yaml
