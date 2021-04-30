#!/bin/bash
#PBS -q v100
#PBS -l nodes=1:ppn=36
#PBS -l walltime=72:00:00
#PBS -A hpc_michal01
#PBS -j oe

cd /work/derick/siamese-monet-project/Pocket2Drug/

singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch171.simg python cross_val_sample.py -fold 0 -result_dir ../p2d_results/cross_val_fold_0/ &> ./cross_validation/sample_logs/sample_fold0.log 2>&1
singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch171.simg python cross_val_sample.py -fold 1 -result_dir ../p2d_results/cross_val_fold_1/ &> ./cross_validation/sample_logs/sample_fold1.log 2>&1
singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch171.simg python cross_val_sample.py -fold 2 -result_dir ../p2d_results/cross_val_fold_2/ &> ./cross_validation/sample_logs/sample_fold2.log 2>&1
singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch171.simg python cross_val_sample.py -fold 3 -result_dir ../p2d_results/cross_val_fold_3/ &> ./cross_validation/sample_logs/sample_fold3.log 2>&1
singularity exec --nv -B /work,/project,/usr/lib64 /work/derick/singularities/pytorch171.simg python cross_val_sample.py -fold 4 -result_dir ../p2d_results/cross_val_fold_4/ &> ./cross_validation/sample_logs/sample_fold4.log 2>&1

