"""
Compute the ligand efficiency according to the docking scores and molecular weights.
"""
import argparse
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import yaml
from openbabel import openbabel

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-docking_score_dir",
                        required=False,
                        default="../../../p2d_results_selfie/cv_results/cross_val_fold_0/zinc_sampled_docking_results/",
                        help="directory of docking scores of molecules generated by model")

    parser.add_argument("-mol_dir",
                        required=False,
                        default="../../../p2d_results_selfie/cv_results/cross_val_fold_0/zinc_sampled_pdbqt/",
                        help="directory of the molecules")

    parser.add_argument("-out_dir",
                        required=False,
                        default="../../../p2d_results_selfie/cv_results/cross_val_fold_0/zinc_docking_results_ligand_efficiency/",
                        help="path to save the ligand efficiency")        

    return parser.parse_args()


def compute_mol_weight(pdbqt_file):
    # pdbqt_file = "../../../p2d_results_selfie/cv_results/cross_val_fold_0/val_pockets_sample_clustered_pdbqt/1a3bB00/1a3bB00-1.pdbqt"
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "smi")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, pdbqt_file)   # Open Babel will uncompress automatically
    return mol.GetMolWt()


if __name__ == '__main__':
    args = get_args()
    docking_score_dir = args.docking_score_dir
    mol_dir = args.mol_dir
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    docking_score_files = [f for f in listdir(
                docking_score_dir) if isfile(join(docking_score_dir, f))]

    # each docking file corresponds to a pocket
    for docking_file in tqdm(docking_score_files):
        # pocket name
        pocket = docking_file.split('_')[0]

        # load the dictionary of docking scores, keys are the file names
        # of pdbqt files
        with open(join(docking_score_dir, docking_file), "r") as f:
            docking_scores = yaml.full_load(f)

        ligand_efficiency = {}

        # for each molecule
        for mol in docking_scores:
            # get docking score 
            docking_score = docking_scores[mol]

            # get pdbqt path 
            pdbqt_dir = mol_dir + pocket
            pdbqt_path = os.path.join(pdbqt_dir, mol) # mol is pocket + number.pdbqt, e.g., 4pnuA03-1.pdbqt

            # molecular weight
            mol_weight = compute_mol_weight(pdbqt_path)

            if docking_score is not None and mol_weight:
                # normalize docking score
                normalized = docking_score / mol_weight

                # add normalized score 
                ligand_efficiency[mol] = normalized

        # save the ligand efficiency  
        with open(os.path.join(out_dir, f"{pocket}_ligand_efficiency.yaml"), "w") as f:
            yaml.dump(ligand_efficiency, f)