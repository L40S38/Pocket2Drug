source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python sample.py -batch_size 1024 -num_batches 2 -pocket_dir ../osf_data/pocket-data/ -popsa_dir ../osf_data/protein-data/ -profile_dir ../osf_data/protein-data/ -result_dir ../p2d_results_selfie/cv_tune_pretrained_graphsite_classifier_rm_position/cross_val_fold_0_05122022_0/ -fold 0
