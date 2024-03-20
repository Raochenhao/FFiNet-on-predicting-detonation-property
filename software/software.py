import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import requests
import tkinter as tk
from tkinter import ttk
from torch_geometric.data import Data
from typing import List
from models.FFiNet_model import FFiNetModel
import networkx as nx
from scipy.sparse import coo_matrix
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from typing import Callable, List, Union
from tkinter import StringVar
from tkinter import scrolledtext

model_D = FFiNetModel(
    feature_per_layer=[72,384,384],
    num_heads=8,
    activation=nn.ReLU(),
    pred_hidden_dim=96,
    pred_dropout=0.1,
    pred_layers=1
)
model_P = FFiNetModel(
    feature_per_layer=[72,32,32,32,32],
    num_heads=8,
    activation=nn.PReLU(num_parameters=1),
    pred_hidden_dim=256,
    pred_dropout=0.30000000000000004,
    pred_layers=2
)
model_Q = FFiNetModel(
    feature_per_layer=[72,32,32,32,32],
    num_heads=8,
    activation=nn.PReLU(num_parameters=1),
    pred_hidden_dim=256,
    pred_dropout=0.30000000000000004,
    pred_layers=1
)

model_D.load_state_dict(torch.load('.../saved models/FFiNetModel_D(0).pt', map_location=torch.device('cpu/gpu')))
model_D.eval()
model_P.load_state_dict(torch.load('.../saved models/FFiNetModel_P(0).pt', map_location=torch.device('cpu/gpu')))
model_P.eval()
model_Q.load_state_dict(torch.load('.../saved models/FFiNetModel_Q(0).pt', map_location=torch.device('cpu/gpu')))
model_Q.eval()

Atom = Chem.rdchem.Atom
AtomFeaturesGenerator = Callable[[Union[Atom, str]], np.ndarray]

ATOM_FEATURES_GENERATOR_REGISTRY = {}

def one_hot_encoding(value: int, choices: List) -> List:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def register_atom_features_generator(features_generator_name: str) \
                                    -> Callable[[AtomFeaturesGenerator], AtomFeaturesGenerator]:
    def decorator(features_generator: AtomFeaturesGenerator) -> AtomFeaturesGenerator:
        ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator
    return decorator

def get_features_generator(features_generator_name: str) -> AtomFeaturesGenerator:
    if features_generator_name not in ATOM_FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found.')
    return ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name]

def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(ATOM_FEATURES_GENERATOR_REGISTRY.keys())
@register_atom_features_generator('atom_type')
def atom_type_features_generator(atom: Atom) -> List:
    atom_type_choices = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53]
    atom_type_value = atom.GetAtomicNum()
    return one_hot_encoding(atom_type_value, atom_type_choices)

@register_atom_features_generator('degree')
def degree_features_generator(atom: Atom) -> List:
    degree_choices = list(range(5))
    degree = atom.GetTotalDegree()
    return one_hot_encoding(degree, degree_choices)

@register_atom_features_generator('chiral_tag')
def chiral_tag_features_generator(atom: Atom) -> List:
    chiral_tag_choices = list(range(len(Chem.ChiralType.names)-1))
    chiral_tag = atom.GetChiralTag()
    return one_hot_encoding(chiral_tag, chiral_tag_choices)

@register_atom_features_generator('num_Hs')
def num_Hs_features_generator(atom: Atom) -> List:
    num_Hs_choices = list(range(5))
    num_Hs = atom.GetTotalNumHs()
    return one_hot_encoding(num_Hs, num_Hs_choices)

@register_atom_features_generator('hybridization')
def hybridization_features_generator(atom: Atom) -> List:
    hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
    hybridization = int(atom.GetHybridization())
    return one_hot_encoding(hybridization, hybridization_choices)

@register_atom_features_generator('aromatic')
def aromatic_features_generator(atom: Atom) -> List:
    return [1 if atom.GetIsAromatic() else 0]

@register_atom_features_generator('mass')
def mass_features_generator(atom: Atom) -> List:
    return [atom.GetMass()]

@register_atom_features_generator('hydrogen_bond')
def hydrogen_bond_features_generator(mol: Chem.Mol, atom: Atom) -> List:
    h_bond_infos = construct_hydrogen_bonding_info(mol)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    return acceptor_donor

def atom_features_union(mol: Chem.Mol, atom: Atom, generator_name_list: List) -> np.ndarray:
    atomFeatures = []
    for generator_name in generator_name_list:
        if generator_name in get_available_features_generators():
            if generator_name == 'hydrogen_bond':
                generator = get_features_generator(generator_name)
                atomFeatures += generator(mol, atom)
            else:
                generator = get_features_generator(generator_name)
                atomFeatures += generator(atom)
        else:
            raise KeyError(f'The generator {generator_name} is not in the generator list')
    return np.array(atomFeatures)

def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    features_dict = {}
    num_atoms = mol.GetNumAtoms()
    mol_add_hs = Chem.AddHs(mol)

    # generate conformer by EDKDG method
    AllChem.EmbedMolecule(mol_add_hs, randomSeed=0xf00d)
    try:
        conf = mol_add_hs.GetConformers()[0]
    except IndexError:
        AllChem.EmbedMultipleConfs(mol_add_hs, 50, pruneRmsThresh=0.5)
        try:
            conf = mol_add_hs.GetConformers()[0]
        except IndexError:
            print(f'{Chem.MolToSmiles(mol)}\'s conformer can not be generated')
            conf = None

    if conf != None:
        features_dict['pos'] = conf.GetPositions()[:num_atoms, :]
    else:
        return None

    adj = Chem.GetAdjacencyMatrix(mol)
    coo_adj = coo_matrix(adj)
    features_dict['edge_index'] = [coo_adj.row, coo_adj.col]

    x = []
    z = []
    for atom in mol.GetAtoms():
        atom_generator_name_list = get_available_features_generators()
        x.append(atom_features_union(mol, atom, atom_generator_name_list))
        z.append(atom.GetAtomicNum())
    features_dict['x'] = x
    features_dict['z'] = z
    dtype = torch.float32
    y=0
    data = Data(
        z=torch.tensor(features_dict['z'], dtype=torch.long),
        x=torch.tensor(features_dict['x'], dtype=dtype),
        edge_index=torch.tensor(features_dict['edge_index'], dtype=torch.long),
        pos=torch.tensor(features_dict['pos'], dtype=dtype),
        y=torch.tensor(y, dtype=dtype))
    adj = Chem.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    data.triple_index = subgraph_index(G, 2)
    data.quadra_index = subgraph_index(G, 3)
    data.smiles = Chem.MolToSmiles(mol)

    return data


def get_smiles_from_identifier(identifier):
    url = f'https://cactus.nci.nih.gov/chemical/structure/{identifier}/smiles'
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None

def predict_smiles(smiles):
    data = smiles_to_data(smiles)
    if data is None:
        return "Invalid SMILES input"
    y_std_D = torch.tensor([[0.8738]])
    y_mean_D = torch.tensor([[7.8842]])
    y_std_P = torch.tensor([[6.0824]])
    y_mean_P = torch.tensor([[28.2008]])
    y_std_Q = torch.tensor([[0.7665]])
    y_mean_Q = torch.tensor([[5.4059]])

    prediction_D = (model_D(data).detach().reshape(-1, 1) * y_std_D + y_mean_D).tolist()
    prediction_P = (model_P(data).detach().reshape(-1, 1) * y_std_P + y_mean_P).tolist()
    prediction_Q = (model_Q(data).detach().reshape(-1, 1) * y_std_Q + y_mean_Q).tolist()

    return prediction_D[0][0],prediction_P[0][0],prediction_Q[0][0]

def subgraph_index(G, n):

    allpaths = []
    for node in G:
        paths = findPaths(G, node , n)
        allpaths.extend(paths)
    allpaths = torch.tensor(allpaths, dtype=torch.long).T
    return allpaths

def findPaths(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths



def on_predict():
    smiles_input = entry_smiles.get()
    if input_type_var.get() == "SMILES":
        smiles = smiles_input
    else:
        smiles = get_smiles_from_identifier(smiles_input)
    if smiles is not None:
        try:
            result_D, result_P, result_Q = predict_smiles(smiles)
            label_result.config(text=f"Predicted Detonation Velocity(km/s): {result_D:.3f}\nPredicted Detonation Pressure(GPa): {result_P:.3f}\nPredicted heat of explosion(KJ/g): {result_Q:.3f}")
        except Exception as e:
            label_result.config(text=f"Prediction error: {str(e)}")
    else:
        label_result.config(text="Invalid molecule identifier or unable to retrieve SMILES.")




window = tk.Tk()
window.title("Molecule Property Predictor")

frame = tk.Frame(window, padx=40, pady=40)
frame.grid(row=0, column=0, columnspan=2)

label_instruction = tk.Label(frame, text="Enter Molecule Identifier:")
label_instruction.grid(row=0, column=0, padx=40, pady=40)

entry_smiles = tk.Entry(frame)
entry_smiles.grid(row=0, column=1, padx=10, pady=10)

button_predict = tk.Button(frame, text="Predict", command=on_predict, bg="#4CAF50", fg="white")
button_predict.grid(row=1, column=0, columnspan=2, pady=10)

label_result = tk.Label(window, text="")
label_result.grid(row=1, column=0, columnspan=2, pady=10)

input_type_var = StringVar()
input_type_var.set("SMILES")

input_type_menu = ttk.Combobox(frame, textvariable=input_type_var, values=["SMILES", "Other Identifier"])
input_type_menu.grid(row=0, column=2, padx=10, pady=10)

window.mainloop()

