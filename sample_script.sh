#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python sample.py -batch_size 1024 -num_batches 2 -pocket_dir ../osf_data/pocket-data/ -pocket_folds_dir ./data/folds-reduced/ -popsa_dir ../osf_data/protein-data/ -profile_dir ../osf_data/protein-data/ -result_dir ../p2d_results_selfie/cv_tune_pretrained_graphsite_classifier/cross_val_fold_0_11252023_1/ -fold 0
