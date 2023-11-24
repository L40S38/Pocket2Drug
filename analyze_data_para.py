from tqdm import tqdm
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem,PandasTools,QED,Descriptors,rdMolDescriptors,rdShapeHelpers,rdMolAlign
import pandas as pd
import glob
import yaml
from rdkit_contrib import sascorer
import numpy as np
from openbabel import pybel
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor

val_pocket_sample_dir = "../p2d_results_selfie/cv_tune_pretrained_gnn/cross_val_fold_0_05122022_0/val_pockets_sample_2048/"

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

if __name__=='__main__':
    gen_yamls = glob.glob(val_pocket_sample_dir + "*.yaml")

    with open('./data/pocket-smiles.yaml', 'r') as yml:
        config = yaml.full_load(yml)

    dfs = []

    for gen_yaml in tqdm(gen_yamls):
        with open(gen_yaml,'r') as yml:
            gen_data = yaml.full_load(yml)
        pocket_name = gen_yaml.split('/')[-1][:7]
        
        # smiles save
        target_smiles = config[pocket_name]
        generated_smiles = [[smiles,n] for smiles,n in gen_data.items()]
        #df_concat = pd.DataFrame([[pocket_name,target_smiles]+generated_smiles[i] for i in range(len(generated_smiles))],columns=['pocket_id','target_smiles','generated_smiles','generated_num'])   

        # mol from smiles
        target_mol = Chem.MolFromSmiles(config[pocket_name])
        fpgen = AllChem.GetRDKitFPGenerator()
        target_fp_rdkit = fpgen.GetFingerprint(target_mol)
        target_fp_morgan = AllChem.GetMorganFingerprintAsBitVect(target_mol,2,1024)
        #fp1 = f"~/osf_data/pocket-data/{pocket_name}/{pocket_name}.mol2"
        fp1 = tempfile.gettempdir()+"/"+f"{pocket_name}_label.mol2"
        subp = subprocess.Popen(f"echo \"{target_smiles}\" | obabel -i smi -o mol2 --gen3D > {fp1}",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,executable='/bin/bash')
        subp.wait(timeout=1)
        data_sub = []
        #for i, (smiles, n) in enumerate(generated_smiles):

        def get_analyze_data(inp):
            i,(smiles,n) = inp
            similarity_morgan = np.nan
            similarity_rdkit = np.nan
            similarity_3d = np.nan
            sascore = np.nan
            qed = np.nan
            rof = np.nan
            fp2 = tempfile.gettempdir()+"/"+f"{pocket_name}_gene{i}.mol2"

            try:
                generated_mol = Chem.MolFromSmiles(smiles)
                #print("smiles to mol")

                #sascore
                sascore = sascorer.calculateScore(generated_mol)
                #print("calculate sascore")

                #QED
                qed = Chem.QED.qed(generated_mol)
                #print("QED calculate")

                #rule-of-five
                rof = rule_of_five(generated_mol)
                #print("rule of five")

                # tanimoto similarity of rdkit fingerprint from mol
                generated_fp = fpgen.GetFingerprint(generated_mol)
                similarity_rdkit = DataStructs.TanimotoSimilarity(target_fp_rdkit,generated_fp)
                #print("rdkit tanimoto")

                # tanimoto similarity of Morgan fingerprint from mol
                generated_fp = AllChem.GetMorganFingerprintAsBitVect(generated_mol,2,1024)
                similarity_morgan= DataStructs.TanimotoSimilarity(target_fp_morgan,generated_fp)
                #print("morgan tanimoto")

                # 3d tanimoto similarity ref:https://magattaca.hatenablog.com/entry/2018/12/17/002524
                #generated_mol_align = rdMolAlign.AlignMol(target_mol,generated_mol)
                #similarity_3d = rdShapeHelpers.ShapeTanimotoDist(target_mol,generated_mol_align)
                #print("3d tanimoto")
                #generated_mol2 = pybel.readstring("smi",smiles).write("mol2")
                subp = subprocess.Popen(f"echo \"{smiles}\" | obabel -i smi -o mol2 --gen3D > {fp2}",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,executable='/bin/bash')
                subp.wait(timeout=1)
                #print(fp1,fp2)
                proc = subprocess.run(f"pkcombu -A {fp1} -B {fp2}", shell=True, text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,executable='/bin/bash')
                #proc.wait(timeout=1)
                lines = proc.stdout.split('\n')
                for j,line in enumerate(lines):
                #while True:
                #    line = proc.stdout.readline()
                    spl = line.split()
                    if len(spl)!=2:
                        continue
                    #print(j,spl)
                    #print(line)
                    if "tanimoto" in spl[0]:
                        print(f"TC={spl[1]}")
                        similarity_3d = float(spl[1])
                        #print(target_smiles,smiles,tanimoto_similarity)
                        break
                #target_molとgenerated_molをmol2ファイルに変換してランダム?なtmpディレクトリに保存（あとでDeeplyToughを参照）
                #その2つについてpkcombuした結果をsubprocessのPIPEでとってきて、tanimoto係数の箇所だけ取得

            except:
                smiles = smiles #do nothing
            
            if os.path.exists(fp2):
                os.remove(fp2)
            
            return [pocket_name,target_smiles,smiles,n,similarity_rdkit,similarity_morgan,similarity_3d,sascore,qed,rof]
        
        with ProcessPoolExecutor() as executor:
            for data in executor.map(get_analyze_data,enumerate(generated_smiles)):
                data_sub.append(data)
                
        df_concat = pd.DataFrame(data_sub,columns=['pocket_id','target_smiles','generated_smiles','generated_num','rdkit_tanimoto','morgan_tanimoto','3d_tanimoto','sa_score','QED','rule_of_five'])
    
        dfs.append(df_concat)
        if os.path.exists(fp1):
            os.remove(fp1)
    df = pd.concat(dfs,ignore_index=True)
    df.to_csv(val_pocket_sample_dir + "all_results_2.csv",encoding='utf-8')
