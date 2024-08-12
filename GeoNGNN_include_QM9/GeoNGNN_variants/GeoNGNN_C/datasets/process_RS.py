from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType
from rdkit import RDLogger
import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="~/datasets/chiral")
args = parser.parse_args()

raw_path = f"{args.data_dir}/raw/rs/"
processed_path = f"{args.data_dir}/processed/rs/"

RDLogger.DisableLog('rdApp.*')

CHIRALTAG_PARITY = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    ChiralType.CHI_UNSPECIFIED: 3,
    ChiralType.CHI_OTHER: 0, # default
}

atomTypesT = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formalChargeT = [-1, -2, 1, 2, 0]
degreeT = [0, 1, 2, 3, 4, 5, 6]
num_HsT = [0, 1, 2, 3, 4]
local_chiral_tagsT = [0, 1, 2, 3]
stereoTypesT = [0, 1, 2, 3, 4, 5]
hybridizationT = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
bondTypesT = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


def get_index(value, options):
    index = options.index(value) if value in options else len(options)
    return index


def getMol(id, smiles, raw_mol, label):
    data = Data()
    data["name"] = [smiles, id, 0]
    pos = []
    z = []
    mass_list = []

    global_tags = dict(rdkit.Chem.FindMolChiralCenters(raw_mol, force=True, includeUnassigned=True, useLegacyImplementation=False))
    atoms = raw_mol.GetAtoms()
    conformer = raw_mol.GetConformer()
    for i, atom in enumerate(atoms):
        p = list(conformer.GetAtomPosition(atom.GetIdx()))
        assert atom.GetIdx() == i
        t = get_index(atom.GetSymbol(), atomTypesT) # 13
        tag = CHIRALTAG_PARITY[atom.GetChiralTag()] # 4
        
        degree = get_index(atom.GetTotalDegree(), degreeT)  # 8
        charge = get_index(atom.GetFormalCharge(), formalChargeT) # 6
        numHs = get_index(atom.GetTotalNumHs(), num_HsT)    # 6
        hybrid = get_index(atom.GetHybridization(), hybridizationT)    # 8
        aromatic = int(atom.GetIsAromatic()) # 2
        mass = atom.GetMass() * 0.01
        
        global_chiral_tag = 0
        if i in global_tags:
            if global_tags[i] == 'R':
                global_chiral_tag = 1
            elif global_tags[i] == 'S':
                global_chiral_tag = 2
            else:
                global_chiral_tag = 3

        pos.append(p)
        z.append([t, tag, degree, charge, numHs, hybrid, aromatic, global_chiral_tag])
        mass_list.append(mass)
    
    
    data["pos"] = torch.tensor(pos, dtype=torch.float)
    data["z"] = torch.tensor(z, dtype=torch.long)
    data["mass"] = torch.tensor(mass_list, dtype=torch.float)
    data["y"] = torch.tensor([float(label)], dtype=torch.float)
    
    N = data["pos"].shape[0]
    edge = torch.zeros((N, N, 4), dtype=torch.long)
    
    bonds = raw_mol.GetBonds()
    for bond in bonds:
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        btype = get_index(str(bond.GetBondType()), bondTypesT) + 1 # 6
        conjugated = int(bond.GetIsConjugated()) + 1    # 3
        ring = int(bond.IsInRing()) + 1     # 3
        stereo = get_index(bond.GetStereo(), stereoTypesT) + 1  # 8
        edge[u, v] = edge[v, u] = torch.Tensor([btype, conjugated, ring, stereo])


    data["edge_matrix"] = edge
    
    return data


import numpy as np

full_dataset = []


train_dataset = "train_RS_classification_enantiomers_MOL_326865_55084_27542.pkl"
val_dataset = "validation_RS_classification_enantiomers_MOL_70099_11748_5874.pkl"
test_dataset = "test_RS_classification_enantiomers_MOL_69719_11680_5840.pkl"


train_path = raw_path + train_dataset
val_path = raw_path + val_dataset
test_path = raw_path + test_dataset

split_name = ["training", "validation", "test"]

for _, path in enumerate([train_path, val_path, test_path]):
    with open(path, 'rb') as fl:
        data = pickle.load(fl)
        data = data.sort_values('SMILES_nostereo').reset_index(drop = True)
        n = data.shape[0]
        for i in tqdm(range(n), "processing " + split_name[_] + " split"):
            line = data.iloc[i]
            mol = getMol(line['SMILES_nostereo'], line['ID'], line['rdkit_mol_cistrans_stereo'], line['RS_label_binary'])
            full_dataset.append(mol)
import os
if os.path.exists(processed_path) == False:
    os.makedirs(processed_path)
torch.save(full_dataset, processed_path + "rs_full.pt")
