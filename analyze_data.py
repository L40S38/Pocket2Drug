from tqdm import tqdm
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem,PandasTools,QED,Descriptors,rdMolDescriptors
import pandas as pd
import glob
import yaml
from rdkit_contrib import sascorer

gen_yamls = glob.glob('../p2d_results_selfie/cv_tune_pretrained_gnn/cross_val_fold_0_05122022_0/val_pockets_sample_2048/*.yaml')

with open('./data/pocket-smiles.yaml', 'r') as yml:
    config = yaml.full_load(yml)

dfs = []

def rule_of_five(mol):
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    psa = Descriptors.TPSA(mol)

    if (mol_weight <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
        # 上の条件式に5の倍数が多く出てくるのでrule of fiveと呼ばれている
        return 1
    else:
        return 0

for gen_yaml in tqdm(gen_yamls):
    with open(gen_yaml,'r') as yml:
        gen_data = yaml.full_load(yml)
    pocket_name = gen_yaml.split('/')[-1][:7]
    # mol from smiles
    target_smiles = config[pocket_name]
    target_mol = Chem.MolFromSmiles(config[pocket_name])
    generated_smiles = [[smiles,n] for smiles,n in gen_data.items()]
    generated_mols = [Chem.MolFromSmiles(smiles) for smiles,n in gen_data.items()]
    df_concat = pd.DataFrame([[pocket_name,target_smiles]+generated_smiles[i] for i in range(len(generated_smiles))],columns=['pocket_id','target_smiles','generated_smiles','generated_num'])   

    # tanimoto similarity fingerprint from mol
    fpgen = AllChem.GetRDKitFPGenerator()
    target_fp = fpgen.GetFingerprint(target_mol)
    generated_fps = [fpgen.GetFingerprint(x) for x in generated_mols]
    similarity = [DataStructs.TanimotoSimilarity(target_fp,generated_fp) for generated_fp in generated_fps]
    df_concat['similarity'] = similarity
        
    dfs.append(df_concat)

df = pd.concat(dfs,ignore_index=True)
df.to_csv('../p2d_results_selfie/cv_tune_pretrained_gnn/cross_val_fold_0_05122022_0/val_pockets_sample_2048/all_results.csv',encoding='utf-8')
