#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python sample.py -batch_size 1024 -num_batches 2 -pocket_dir ~/add_val_data/POCKETS_TRANSFORMED_MOL2/ -popsa_dir ~/add_val_data/PDB_CHAINS/ -profile_dir ~/add_val_data/PDB_CHAINS/ -result_dir ~/add_val_data/
