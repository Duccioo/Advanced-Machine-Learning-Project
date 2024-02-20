import argparse
import os

import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR


import torch_geometric.transforms as T
from torch_geometric.datasets import QM9


# ---
from model_graphvae import GraphVAE
from data_graphvae import create_padded_graph_list, data_to_smiles


def build_model(
    max_num_nodes: int = 0,
    max_num_edges: int = 0,
    num_nodes_features: int = 15,
    num_edges_features: int = 4,
    len_num_features: int = 8,
    latent_dimension=5,
    device=torch.device("cpu"),
):

    # out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    # print(num_features)
    # print(max_num_nodes)

    input_dim = len_num_features
    model = GraphVAE(
        input_dim,
        256,
        latent_dimension,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        device=device,
    ).to(device)
    return model


def train(args, dataloader, model, epoch=50, device=torch.device("cpu")):
    LR_milestones = [500, 1000]
    
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    for epoch in range(epoch):
        for batch_idx, data in enumerate(dataloader, 0):
            

            features_nodes = data["features_nodes"].float().to(device)
            features_edges = data["features_edges"].float().to(device)
            adj_input = data["adj"].float().to(device)
            # print("----------------")
            # print(adj_input)
            # print(features_edges)
            # print(features_nodes)

            model.zero_grad()
            loss = model(adj_input, features_edges, features_nodes)

            loss.backward()
            optimizer.step()
            scheduler.step()

        print("Epoch: ", epoch, ", Loss: ", round(loss.item(), 4))


def count_edges(adj_matrix):
    num_edges = (
        torch.sum(adj_matrix) / 2
    )  # Diviso per 2 perché la matrice adiacente è simmetrica
    return num_edges


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

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
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by training data.",
    )

    parser.add_argument(
        "--num_examples", type=int, dest="num_examples", help="Number of examples"
    )
    parser.add_argument(
        "--latent_dimension", type=int, dest="latent_dimension", help="Latent Dimension"
    )
    parser.add_argument("--epochs", type=int, dest="epochs", help="Number of epochs")
    parser.add_argument("--device", type=str, dest="device", help="cuda or cpu")

    parser.set_defaults(
        dataset="enzymes",
        feature_type="struct",
        lr=0.001,
        batch_size=5,
        num_workers=1,
        max_num_nodes=-1,
        num_examples=20,
        latent_dimension=5,
        epochs=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def main():

    np.random.seed(42)
    torch.manual_seed(42)

    prog_args = arg_parse()

    device = prog_args.device

    # loading dataset
    dataset = QM9(
        root=os.path.join("data", "QM9"),
    )
    dataset = dataset[0 : prog_args.num_examples]

    num_graphs_raw = len(dataset)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([dataset[i].num_nodes for i in range(len(dataset))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        dataset = [g for g in dataset if g.num_nodes <= max_num_nodes]

    graphs_len = len(dataset)

    dataset_padded, max_num_nodes, max_num_edges = create_padded_graph_list(
        dataset,
        prog_args.max_num_nodes,
        add_edge_features=True,
        remove_hidrogen=True,
        one_hot_features_nodes=True,
    )

    print(
        "Number of graphs removed due to upper-limit of number of nodes: ",
        num_graphs_raw - graphs_len,
    )

    # split dataset
    # print(graphs[0])
    # graphs_test = graphs[int(0.8 * graphs_len) :]
    graphs_train = dataset_padded

    print(
        "total graph num: {}, training set: {}".format(
            len(dataset_padded), len(graphs_train)
        )
    )
    print("max number node: {}".format(max_num_nodes))

    # ------ TRAINING -------
    dataset_loader = torch.utils.data.DataLoader(
        graphs_train,
        batch_size=prog_args.batch_size,
        num_workers=prog_args.num_workers,
    )
    print("-------- TRAINING: --------")

    print(max_num_edges)
    print(max_num_nodes)
    print("num edges features", graphs_train[0]["features_edges"].shape[1])
    print("num nodes features", graphs_train[0]["features_nodes"].shape[1])
    model = build_model(
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=graphs_train[0]["features_nodes"].shape[1],
        num_edges_features=graphs_train[0]["features_edges"].shape[1],
        len_num_features=graphs_train[0]["features_nodes"].shape[0],
        latent_dimension=prog_args.latent_dimension,
        device=device,
    )

    train(prog_args, dataset_loader, model, epoch=prog_args.epochs, device=device)

    # ---- INFERENCE ----
    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, prog_args.latent_dimension)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj, features_nodes, features_edges = model.generate(z, device=device)

    rounded_adj_matrix = torch.round(adj)
    # features_nodes = torch.round(features_nodes)
    # features_edges = torch.round(features_edges)
    print("ORIGINAL MATRIX")
    print(graphs_train[0]["adj"])
    print("##" * 10)
    print("Predicted MATRIX")
    print(rounded_adj_matrix)
    print(features_nodes)
    print(features_edges)
    # smiles = data_to_smiles(features_nodes, features_edges, rounded_matrix)
    # print(smiles)

    print("MATCH DELLE MATRICI")
    print(features_edges.shape)
    print(count_edges(rounded_adj_matrix))    
    model.save_vae_encoder("graphvae_modified_v2")


if __name__ == "__main__":
    main()
