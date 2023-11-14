#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda remove -n pocket2drug --all -y
conda create -n pocket2drug -y
conda activate pocket2drug
conda install -c pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia -y 
conda install pyg -c pyg -y
conda install biopandas -c conda-forge -y
pip install selfies
conda install rdkit -c conda-forge -y
conda deactivate > /dev/null
