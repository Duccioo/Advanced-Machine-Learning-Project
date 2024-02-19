import networkx as nx
import numpy as np
import torch

from torch_geometric.utils import to_networkx
import torch_geometric.data
import networkx as nx
import random
import rdkit.Chem as Chem


class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features="id"):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            adj = nx.to_numpy_array(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == "id":
                self.feature_all.append(np.identity(max_num_nodes))
            elif features == "deg":
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(
                    np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0), axis=1
                )
                self.feature_all.append(degs)
            elif features == "struct":
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(
                    np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], "constant"),
                    axis=1,
                )
                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(
                    np.pad(
                        clusterings,
                        [0, max_num_nodes - G.number_of_nodes()],
                        "constant",
                    ),
                    axis=1,
                )
                self.feature_all.append(np.hstack([degs, clusterings]))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0

        adj_vectorized = adj_padded[
            np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1
        ]
        # the following 2 lines recover the upper triangle of the adj matrix
        # recovered = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # recovered[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes)) ) == 1] = adj_vectorized
        # print(recovered)

        return {
            "adj": adj_padded,
            "adj_decoded": adj_vectorized,
            "features": self.feature_all[idx].copy(),
        }


def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()[0]

    # print(y)

    # print(n_sample, g.nodes)

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y


def create_adj_list(dataset):
    adj_list = []
    for data in dataset:
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        G = nx.Graph(edge_list)
        adj = nx.adjacency_matrix(G)
        adj_list.append({"adj": np.array(adj.todense())})
    return adj_list


def create_graph_list(dataset):
    graph_list = []
    for data in dataset:
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        G = nx.Graph(edge_list)
        adj = nx.adjacency_matrix(G)
        graph = {
            "adj": np.array(adj.todense()),
            "features": np.array(data.x),  # Aggiunta delle features corrispondenti
        }
        graph_list.append(graph)
    return graph_list


def pad_adjacency_matrix(adj, max_nodes):
    pad_size = max_nodes - adj.shape[0]
    if pad_size < 0:
        pad_size = 0
    padded_adj = np.pad(adj, ((0, pad_size), (0, pad_size)), mode="constant")
    return padded_adj


def remove_hidrogen_from_dataset(
    dataset, save_to_file: str = False, remove_edge_attr_duplicates=True
):
    # Filtra gli atomi di idrogeno
    filtered_data = []
    for data in dataset:
        # Verifica se l'atomo è un idrogeno (Z=1)
        is_hydrogen = data.z == 1

        # Filtra gli edge e gli edge_index
        edge_mask = ~is_hydrogen[data.edge_index[0]] & ~is_hydrogen[data.edge_index[1]]
        data.edge_index = data.edge_index[:, edge_mask]

        # Filtra le features dei nodi
        data.x = data.x[~is_hydrogen]
        data.edge_attr = data.edge_attr[edge_mask]

        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        if edge_list:
            G = nx.Graph(edge_list)
            adj = nx.adjacency_matrix(G).todense()
            # print("e la matrice adiacenze è....", adj)

        data.z = data.z[~is_hydrogen]

        data.num_nodes = len(data.x)
        data.num_edges = len(data.edge_attr) // 2

        # rimuovo dagli attributi i duplicati
        # if remove_edge_attr_duplicates:
            
        #     upper_triangular_indices = torch.triu_indices(
        #         row=adj[i].size(0), col=adj[i].size(1), offset=1
        #     )

        #     # Estraiamo gli elementi corrispondenti dalla matrice
        #     adj_wout_diagonal = adj[i][
        #         upper_triangular_indices[0], upper_triangular_indices[1]
        #     ]
        #     # adj_wout_diagonal = adj[i].triu(diagonal=1).flatten()
        #     adj_mask = adj_wout_diagonal.repeat(4, 1).T

        #     print("ADJ without diagonal ", adj_wout_diagonal)

        #     print("MASCHERA = ", adj_mask)
        #     print("RECON ADJ", edges_recon_features[i])
        #     masked_edges_recon_features = edges_recon_features[i] * adj_mask
        #     print("MASCHERATO =  ", masked_edges_recon_features)

        if data.num_edges != 0 and data.num_nodes != 1:
            # Aggiungi il grafo filtrato alla lista
            filtered_data.append(data)

    if save_to_file != False:
        torch_geometric.data.InMemoryDataset(root=save_to_file, data_list=filtered_data)
    return filtered_data


def create_padded_graph_list(
    dataset,
    max_num_nodes_padded=-1,
    add_edge_features: bool = False,
    one_hot_features_nodes=False,
    remove_hidrogen=True,
):
    if remove_hidrogen:
        # rimuovo gli atomi di idrogeno
        dataset = remove_hidrogen_from_dataset(dataset)

    if max_num_nodes_padded == -1:
        max_num_nodes = max([data.num_nodes for data in dataset])
    else:
        max_num_nodes = max_num_nodes_padded

    # max_num_edges = max([data.num_edges for data in dataset])

    # numero massimo teorico per un graph
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    graph_list = []
    for data in dataset:
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        if edge_list:
            G = nx.Graph(edge_list)
            adj = nx.adjacency_matrix(G).todense()
            padded_adj = pad_adjacency_matrix(adj, max_num_nodes)

        else:
            padded_adj = np.zeros((max_num_nodes, max_num_nodes))

        if max_num_nodes - data.num_nodes < 0:
            continue

        if one_hot_features_nodes:
            # Indice della colonna da codificare
            col_index = 5
            # Otteniamo la colonna da codificare
            col_to_encode = data.x[:, col_index]

            # Calcoliamo il numero di classi uniche nella colonna

            # Applichiamo one-hot encoding utilizzando la funzione di numpy
            # Dizionario di mapping specificato
            mapping_dict = {
                6: [1, 0, 0, 0],
                7: [0, 1, 0, 0],
                8: [0, 0, 1, 0],
                9: [0, 0, 0, 1],
            }
            # Applichiamo il mapping utilizzando il metodo map di Python
            one_hot_encoded = torch.tensor(
                list(map(lambda x: mapping_dict[x.item()], col_to_encode))
            )

            # Sostituiamo la quinta colonna con la codifica one-hot
            # Sostituiamo la quinta colonna con la codifica one-hot
            data.x = torch.cat(
                (
                    data.x[:, :col_index],
                    one_hot_encoded,
                    data.x[:, col_index + 1 :],
                ),
                dim=1,
            )

        features_nodes_padded = torch.cat(
            [
                data.x,
                torch.zeros(max_num_nodes - data.num_nodes, data.num_features),
            ]
        )
        features_edge_padded = data.edge_attr
        if add_edge_features:
            # print("----------")
            # print(padded_adj)
            # # print(data.x)
            # print(data.edge_attr)
            # print(max_num_edges)
            # print(data.num_edges)
            features_edge_padded = torch.cat(
                [
                    data.edge_attr,
                    torch.zeros(
                        max_num_edges * 2 - data.num_edges * 2, data.edge_attr.shape[1]
                    ),
                ]
            )

        graph = {
            "adj": np.array(padded_adj, dtype=np.float32),
            "features_nodes": features_nodes_padded,  # Aggiunta delle features con padding
            "features_edges": features_edge_padded,
            "num_nodes": len(data.z),
            "num_edges": data.num_edges,
        }

        # print(graph)

        graph_list.append(graph)

        print(graph["features_edges"].shape)
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
