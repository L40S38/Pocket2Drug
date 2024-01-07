#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python analyze_data_para2.py -val_pocket_sample_dir ~/p2d_results_selfie/cv_tune_pretrained_graphsite_classifier_rm_position/cross_val_fold_0_01072024_non_trained/val_pockets_sample_2048/ -config_yaml ./data/pocket-smiles-reduced.yaml
