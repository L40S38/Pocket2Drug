# Pocket2Drug
Pocket2Drug is an encoder-decoder deep neural network that predicts binding drugs given protein binding sites (pockets). The pocket graphs are generated using [Graphsite](https://github.com/shiwentao00/Graphsite). The encoder is a graph neural network, and the decoder is a recurrent neural network. The [SELFIES](https://github.com/aspuru-guzik-group/selfies) molecule representation is used as the tokenization scheme instead of SMILES. The pipeline of Pocket2Drug is illustrated below:
<p align="center">
<img width="820" height="260" src="doc/pipeline.png">
</p>

If you find this work helpful, please cite our paper:  
Shi, Wentao, et al. "Pocket2Drug: An encoder-decoder deep neural network for the target-based drug design." Frontiers in Pharmacology: 587.

## Usage
### Dependencies
Pocket2Drug is implemented based on the folloing Python packages:
1. [Pytorch](https://pytorch.org/get-started/locally/)
2. [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
3. [Rdkit](https://www.rdkit.org/docs/Install.html)
4. [selfies](https://github.com/aspuru-guzik-group/selfies)
6. [BioPandas](http://rasbt.github.io/biopandas/)

### Installation
1. Install Pytorch
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

2. Install Pytorch geometric:
```
conda install pyg -c pyg
```

3. Install Biopandas:
```
conda install biopandas -c conda-forge
```

4. Install SELFIES:
```
pip install selfies
```

### Dataset
All the related data can be downloaded [here](). There are two dataset files:
1. dataset.tar.gz: contains all binding site data in this project.
2. pops.tar.gz: contains information of node feature contact surface area.

### Configuration

### Train
The configurations for training can be updated in ```train.yaml```. Modify the ```pocket_dir``` and ```pop_dir``` entries to the paths of the extracted dataset. Modify the ```out_dir``` entry to the folder where you want to save the output results. The other configurations are for hyper-parameter tuning and they are self-explanatory according to their names.
```
python train.py
```
Note: the results presented in our research paper were produced with Pytorch version 1.7.1 and selfies version v1.0, which are far behind current version. There has been a major update for selfies, so the generated token vocabulary of the molecules has changed and the pre-trained RNN can not be used for training (The pre-trained RNN learns the distribution of the chembl database, which improves the performance of the model. I have wrote an exmaple [here](https://github.com/shiwentao00/Molecule-RNN)). As a result, the performance of the model will be effected. I will update the configuration once I get a chance to re-do the pretraining and hyper-parameter tuning.

### Inference
After training, the trained model will be saved at ```out_dir```, and we can use it to sample predicted molecules:
```
python sample.py -batch_size 1024 -num_batches 1 -pocket_dir path_to_dataset_folder -popsa_dir path_to_pops_folder
```

### Inference using the trained model
