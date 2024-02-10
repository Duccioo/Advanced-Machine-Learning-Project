import argparse
import networkx as nx
import os
import random

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx
import rdkit.Chem as Chem

# ---
import data
from model_graphvae import GraphVAE

# from data_graphvae import GraphAdjSampler


CUDA = 2

LR_milestones = [500, 1000]


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
    padded_adj = np.pad(adj, ((0, pad_size), (0, pad_size)), mode="constant")
    return padded_adj


def create_padded_graph_list(
    dataset, max_num_nodes_padded=-1, add_edge_features: bool = False
):
    if max_num_nodes_padded == -1:
        max_num_nodes = max([data.num_nodes for data in dataset])
    else:
        max_num_nodes = max_num_nodes_padded

    max_num_edges = max([data.num_edges for data in dataset])

    graph_list = []
    for data in dataset:
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        G = nx.Graph(edge_list)
        adj = nx.adjacency_matrix(G).todense()
        padded_adj = pad_adjacency_matrix(adj, max_num_nodes)
        # print(data)
        # print("Nodes Features", data.x)
        # print("Edge Features", data.edge_attr)
        # print("----------")
        # print(max_num_edges)
        # print(max_num_nodes)
        # print(data.num_nodes)
        # print(data.edge_attr.shape[0])
        if add_edge_features:
            padded_dimension_node = (
                max_num_nodes - data.num_nodes
                if max_num_edges < data.num_nodes
                else max_num_edges - data.num_nodes
            )
            # print("Padded dimension node", padded_dimension_node)

            padded_dimension_edge = (
                max_num_nodes - max_num_edges
                if max_num_edges < data.num_nodes
                else max_num_edges - data.num_edges
            )

            # print("Padded dimension edge", padded_dimension_edge)

            features_array = np.concatenate(
                (
                    torch.cat(
                        [
                            data.x,
                            torch.zeros(padded_dimension_node, data.num_features),
                        ]
                    ),
                    torch.cat(
                        [
                            data.edge_attr,
                            torch.zeros(
                                padded_dimension_edge,
                                data.edge_attr.shape[1],
                            ),
                        ]
                    ),
                ),
                axis=1,
            )
        else:
            features_array = torch.cat(
                [
                    data.x,
                    torch.zeros(max_num_nodes - data.num_nodes, data.num_features),
                ]
            )

        graph = {
            "adj": np.array(padded_adj),
            "features": features_array,  # Aggiunta delle features con padding
            "num_nodes": len(data.z),
        }
        graph_list.append(graph)
    return graph_list


def build_model(
    max_num_nodes: int = 0,
    num_features: int = 15,
    len_num_features: int = 8,
    latent_dimension=5,
):

    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    # print(num_features)
    # print(max_num_nodes)

    input_dim = len_num_features
    model = GraphVAE(
        input_dim, 256, latent_dimension, max_num_nodes, num_features=num_features
    )
    return model


def train(args, dataloader, model, epoch=50):
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    iteration_all = 0
    for epoch in range(epoch):
        loss = 0
        model.zero_grad()

        for batch_idx, data in enumerate(dataloader):
            iteration_all += 1
            # print(data["adj"].shape)
            # print(data["features"].shape)
            features = data["features"].float()
            adj_input = data["adj"].float()

            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()

            # print(features)

            # print("Calcolo la Loss")
            loss = model(features, adj_input)

            loss.backward()
            optimizer.step()
            scheduler.step()

        print("Epoch: ", epoch, ", Loss: ", loss.item())
    print(iteration_all)


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )

    parser.set_defaults(
        dataset="enzymes",
        feature_type="struct",
        lr=0.001,
        batch_size=1,
        num_workers=1,
        max_num_nodes=-1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA)
    print("CUDA", CUDA)
    ### running log

    # # loading dataset
    dataset = QM9(root="data/QM9")

    NUM_EXAMPLES = 5
    LATENT_DIMENSION = 5
    dataset_padded = create_padded_graph_list(
        dataset[0:NUM_EXAMPLES], prog_args.max_num_nodes, add_edge_features=False
    )
    num_graphs_raw = len(dataset_padded)

    if prog_args.max_num_nodes == -1:
        graphs = dataset_padded
        max_num_nodes = max([graphs[i]["num_nodes"] for i in range(len(graphs))])
        # max_num_nodes = len(graphs[0]["features"])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in dataset_padded if g["num_nodes"] <= max_num_nodes]

    graphs_len = len(graphs)
    print(
        "Number of graphs removed due to upper-limit of number of nodes: ",
        num_graphs_raw - graphs_len,
    )

    # split dataset
    print("-----------")
    print(graphs[0])
    graphs_test = graphs[int(0.8 * graphs_len) :]
    graphs_train = graphs

    print(graphs_train[0])

    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(max_num_nodes))

    # ------

    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size,
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
        graphs_train,
        batch_size=prog_args.batch_size,
        num_workers=prog_args.num_workers,
    )
    model = build_model(
        max_num_nodes=max_num_nodes,
        num_features=graphs_train[0]["features"].shape[1],
        len_num_features=graphs_train[0]["features"].shape[0],
        latent_dimension=LATENT_DIMENSION,
    ).cuda()
    train(prog_args, dataset_loader, model, epoch=1)

    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, LATENT_DIMENSION)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj, features_nodes, features_edges = model.generate(z, device="cuda")

    rounded_matrix = torch.round(adj)
    # features_nodes = torch.round(features_nodes)
    features_edges = torch.round(features_edges)
    print(rounded_matrix)
    print(features_nodes)
    print(features_edges)
    smiles = data_to_smiles(features_nodes, features_edges, rounded_matrix)
    print(smiles)


def calc_metrics(
    smiles_true: list = ["CCO", "CCN", "CCO", "CCC"],
    smiles_predicted: list = ["CCO", "CCN", "CCC", "CCF"],
):
    # Esempio di utilizzo:
    generated_molecules = smiles_predicted
    training_set_molecules = smiles_true

    validity_percentage = sum(
        calculate_validity(mol) for mol in generated_molecules
    ) / len(generated_molecules)
    uniqueness_percentage = calculate_uniqueness(generated_molecules)
    novelty_percentage = calculate_novelty(generated_molecules, training_set_molecules)

    print(f"Validità: {validity_percentage:.2%}")
    print(f"Unicità: {uniqueness_percentage:.2%}")
    print(f"Novità: {novelty_percentage:.2%}")


def calculate_validity(molecule):
    try:
        mol = Chem.MolFromSmiles(molecule)
        return mol is not None
    except:
        return False


def calculate_uniqueness(molecules):
    unique_molecules = set()
    for mol in molecules:
        unique_molecules.add(mol)
    return len(unique_molecules) / len(molecules)


def calculate_novelty(generated_molecules, training_set_molecules):
    novel_molecules = set(generated_molecules) - set(training_set_molecules)
    return len(novel_molecules) / len(generated_molecules)


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


if __name__ == "__main__":
    main()
