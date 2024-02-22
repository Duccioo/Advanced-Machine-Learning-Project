import networkx as nx
import numpy as np
import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj

import rdkit.Chem as Chem


class ToTensor(BaseTransform):
    def __call__(self, data):
        data.x = torch.tensor(data.x)
        data.y = np.array(data.y)
        data.adj = torch.tensor(data.adj)
        return data


class CostumPad(BaseTransform):
    def __init__(self, max_num_nodes, max_num_edges):
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

    def __call__(self, data):
        data.adj = pad_adjacency_matrix(data.adj, self.max_num_nodes)
        data.edge_attr = pad_features(data.edge_attr, self.max_num_edges)
        data.x = pad_features(data.x, self.max_num_nodes)
        data.num_nodes = self.max_num_nodes
        data.num_edges = self.max_num_edges

        return data


class AddAdj(BaseTransform):
    def __call__(self, data):
        # Aggiungere gli adiacenti

        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        if edge_list:
            G = nx.Graph(edge_list)
            adj = nx.adjacency_matrix(G).todense()

        # edge_index è la lista di adiacenza del grafo
        # num_nodes è il numero totale di nodi nel grafo
        # adj_matrix = to_dense_adj(edge_index)

        data.adj = adj

        return data


class OneHotEncoding(BaseTransform):
    def __call__(self, data):
        # Indice della colonna da codificare
        col_index = 5

        # Applichiamo one-hot encoding utilizzando la funzione di numpy
        # Dizionario di mapping specificato
        mapping_dict = {
            1: [0, 0, 0, 0],
            6: [1, 0, 0, 0],
            7: [0, 1, 0, 0],
            8: [0, 0, 1, 0],
            9: [0, 0, 0, 1],
        }

        data.x = one_hot_encoding(data.x, col_index, mapping_dict)

        col_index = 13
        mapping_dict = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1],
            4: [0, 0, 0, 0],
        }

        data.x = one_hot_encoding(data.x, col_index, mapping_dict)

        return data


def one_hot_encoding(matrix, col_index, mapping_dict):

    # One Hot Encoding

    # Otteniamo la colonna da codificare
    col_to_encode = matrix[:, col_index]
    # Applichiamo il mapping utilizzando il metodo map di Python
    one_hot_encoded = torch.tensor(
        list(map(lambda x: mapping_dict[int(x.item())], col_to_encode))
    )

    # Sostituiamo la quinta colonna con la codifica one-hot
    matrix = torch.cat(
        (
            matrix[:, :col_index],
            one_hot_encoded,
            matrix[:, col_index + 1 :],
        ),
        dim=1,
    )

    return matrix


class FilterSingleton(BaseTransform):
    def __call__(self, data):

        data = remove_hydrogen(data)

        if data.x.size(0) == 1 or data == None:
            return False
        else:

            return True


class FilterMaxNodes(BaseTransform):
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes

    def __call__(self, data):
        return self.max_num_nodes == -1 or data.num_nodes <= self.max_num_nodes


def remove_hydrogen(data):

    # Verifica se l'atomo è un idrogeno (Z=1)
    is_hydrogen = data.z == 1

    # Filtra gli edge e gli edge_index
    edge_mask = ~is_hydrogen[data.edge_index[0]] & ~is_hydrogen[data.edge_index[1]]
    data.edge_index = data.edge_index[:, edge_mask]

    # Filtra le features dei nodi
    data.x = data.x[~is_hydrogen]
    data.edge_attr = data.edge_attr[edge_mask]

    data.z = data.z[~is_hydrogen]

    data.num_nodes = len(data.x)
    data.num_edges = len(data.edge_attr)

    return data


def pad_adjacency_matrix(adj_matrix, target_size):
    padded_adj_matrix = torch.nn.functional.pad(
        adj_matrix,
        (0, target_size - adj_matrix.size(0), 0, target_size - adj_matrix.size(1)),
    )
    return padded_adj_matrix


def pad_features(features_matrix, target_size):
    padded_features = torch.nn.functional.pad(
        features_matrix, (0, 0, 0, target_size - features_matrix.size(0))
    )
    return padded_features


def create_padded_graph_list(
    dataset,
    max_num_nodes_padded=-1,
    add_edge_features: bool = False,
    remove_duplicates_features: bool = True,
):

    max_num_nodes = max_num_nodes_padded

    # max_num_edges = max([data.num_edges for data in dataset])

    # numero massimo teorico per un graph
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    graph_list = []
    for data in dataset:

        if remove_duplicates_features:
            # rimuovi duplicati dalla matrice delle features degli edges
            # cerca gli edge unici
            unique_edges = list(
                set(tuple(sorted(x.tolist())) for x in data.edge_index.T)
            )
            # estraggo gli indici degli edge unici
            indices = [
                data.edge_index.T.tolist().index(list(edge)) for edge in unique_edges
            ]
            data.edge_attr = data.edge_attr[indices]

        data.adj = pad_adjacency_matrix(data.adj, max_num_nodes)
        data.edge_attr = pad_features(data.edge_attr, max_num_edges)
        data.x = pad_features(data.x, max_num_nodes)

        graph = {
            "adj": np.array(data.adj, dtype=np.float32),
            "features_nodes": data.x,  # Aggiunta delle features con padding
            "features_edges": data.edge_attr,
            "num_nodes": len(data.z),
            "num_edges": data.num_edges,
        }

        graph_list.append(graph)

    return graph_list, max_num_nodes, max_num_edges


def data_to_smiles(
    node_features, edge_features, adj_matrix, atomic_numbers_idx: int = 5
):
    # Crea un dizionario per mappare i numeri atomici ai simboli atomici
    # atomic_numbers = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    atomic_numbers = {0: "NULL", 1: "H", 2: "C", 3: "N", 4: "O", 5: "F"}

    # Crea un dizionario per mappare la rappresentazione one-hot encoding ai tipi di legami
    bond_types = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
    }

    # Creazione di un elenco di archi dall'adiacenza
    edges_index = torch.nonzero(adj_matrix, as_tuple=False).t()

    # Crea un oggetto molecola vuoto
    mol = Chem.RWMol()

    # Aggiungi gli atomi alla molecola
    print(len(node_features.tolist()))
    number_atom = 0
    for idx, atom in enumerate(node_features.tolist()):
        number_atom += 1
        # atom = mol.AddAtom(Chem.Atom(atomic_numbers[int(atom[atomic_numbers_idx])]))
        # print(data.x)
        print((node_features[idx]))
        if int(node_features[idx][atomic_numbers_idx]) != 0:
            atom_ = Chem.Atom(
                atomic_numbers[int(node_features[idx][atomic_numbers_idx])]
            )
            # print(data.pos[idx, 0])
            # atom_.SetDoubleProp("x", data.pos[idx, 0].item())
            # atom_.SetDoubleProp("y", data.pos[idx, 1].item())
            # atom_.SetDoubleProp("z", data.pos[idx, 2].item())
            mol.AddAtom(atom_)

    # Aggiungi i legami alla molecola
    edge_index = edges_index.tolist()
    print(edge_index)
    bond_saved = []

    for idx, start_end in enumerate(zip(edge_index[0], edge_index[1])):
        start, end = start_end
        # bond_type_one_hot = int((edge_features[idx]).argmax())
        bond_type_one_hot = 0
        bond_type = bond_types[bond_type_one_hot]
        if (
            (start, end) not in bond_saved
            and (
                end,
                start,
            )
            not in bond_saved
            and start != end
        ):
            print("Bound saved, ", start, end, bond_type)
            bond_saved.append((start, end))
            mol.AddBond(start, end, bond_type)

    # Converti la molecola in una stringa SMILES
    smiles = Chem.MolToSmiles(mol)
    return smiles, number_atom


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for bond_ij, atom_i, atom_j in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule


if __name__ == "__main__":

    """
    ### Define helper functions
    These helper functions will help convert SMILES to graphs and graphs to molecule objects.

    **Representing a molecular graph**. Molecules can naturally be expressed as undirected
    graphs `G = (V, E)`, where `V` is a set of vertices (atoms), and `E` a set of edges
    (bonds). As for this implementation, each graph (molecule) will be represented as an
    adjacency tensor `A`, which encodes existence/non-existence of atom-pairs with their
    one-hot encoded bond types stretching an extra dimension, and a feature tensor `H`, which
    for each atom, one-hot encodes its atom type. Notice, as hydrogen atoms can be inferred by
    RDKit, hydrogen atoms are excluded from `A` and `H` for easier modeling.

    """

    atom_mapping = {"C": 0, 0: "C", "N": 1, 1: "N", "O": 2, 2: "O", "F": 3, 3: "F"}

    bond_mapping = {
        "SINGLE": 0,
        0: Chem.BondType.SINGLE,
        "DOUBLE": 1,
        1: Chem.BondType.DOUBLE,
        "TRIPLE": 2,
        2: Chem.BondType.TRIPLE,
        "AROMATIC": 3,
        3: Chem.BondType.AROMATIC,
    }

    NUM_ATOMS = 9  # Maximum number of atoms
    ATOM_DIM = 4 + 1  # Number of atom types
    BOND_DIM = 4 + 1  # Number of bond types
    LATENT_DIM = 64  # Size of the latent space

    # Test helper functions
    graph_to_molecule(smiles_to_graph(smiles))

    """
    ### Generate training set

    To save training time, we'll only use a tenth of the QM9 dataset.
    """

    adjacency_tensor, feature_tensor = [], []
    for smiles in data[::10]:
        adjacency, features = smiles_to_graph(smiles)
        adjacency_tensor.append(adjacency)
        feature_tensor.append(features)

    adjacency_tensor = np.array(adjacency_tensor)
    feature_tensor = np.array(feature_tensor)

    print("adjacency_tensor.shape =", adjacency_tensor.shape)
    print("feature_tensor.shape =", feature_tensor.shape)
