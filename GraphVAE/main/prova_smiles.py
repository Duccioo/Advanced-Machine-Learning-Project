from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
from rdkit.Chem import Draw
import os


def adj_to_smiles_2(adj, node_labels, edge_features, sanitize, cleanup):
    mol = Chem.RWMol()
    smiles = ""

    atomic_numbers = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}

    # Crea un dizionario per mappare la rappresentazione one-hot encoding ai tipi di legami
    bond_types = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
    }

    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(atomic_numbers[node_label]))

    idx = 0
    for start, end in zip(*np.nonzero(adj)):

        if start > end:
            idx += 1
            bond_type_one_hot = int((edge_features[idx]).argmax())
            bond_type = bond_types[bond_type_one_hot]
            try:
                mol.AddBond(int(start), int(end), bond_type)
            except:
                print("ERROR Impossibile aggiungere legame, Molecola incompleta")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol = None

    if cleanup:
        try:
            mol = Chem.AddHs(mol)
            smiles = Chem.MolToSmiles(mol)

            smiles = max(smiles.split("."), key=len)
            if "*" not in smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = None
        except Exception:
            mol = None
    if smiles == "" or smiles == None:
        print("ERROR impossibile creare Molecola")
    return mol, smiles


def adjacency_to_smiles(adjacency_matrix, atomic_numbers):
    # Crea un oggetto molecola vuoto
    molecule = Chem.RWMol()

    # Aggiungi gli atomi alla molecola
    for atomic_number in atomic_numbers:
        atom = Chem.Atom(atomic_number)
        molecule.AddAtom(atom)

    print("e qui ci sono")
    # Aggiungi i legami alla molecola
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            print("DIO MERDA", i, j, adjacency_matrix[i, j])
            if adjacency_matrix[i, j] != 0:
                molecule.AddBond(
                    i, j, Chem.BondType.SINGLE
                )  # Modifica il tipo di legame in base alla matrice dei legami

    # Normalizza la molecola
    # molecule = rdmolops.SanitizeMol(molecule)
    # Converte la molecola in una stringa SMILES
    smiles = Chem.MolToSmiles(molecule)
    molecule = Chem.AddHs(molecule)

    return smiles


def save_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


if __name__ == "__main__":
    # Matrice di adiacenza per H2O
    adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    print(adjacency_matrix)

    # Vettore dei numeri atomici per H2O (1 = Idrogeno, 8 = Ossigeno)
    atomic_numbers = [1, 2, 3]
    edge_features = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

    # Usa la funzione per ottenere la stringa SMILES
    mol, smiles = adj_to_smiles_2(
        adjacency_matrix, atomic_numbers, edge_features, sanitize=True, cleanup=True
    )
    print(mol)
    save_filepath = os.path.join("", "mol_{}.png".format(1))
    save_png(mol, save_filepath, size=(600, 600))
    print(smiles)
