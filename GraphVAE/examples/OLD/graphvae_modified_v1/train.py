import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import random


from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx

# ---
import data
from model_baselines import GraphVAE
from data import GraphAdjSampler

CUDA = 1

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


def show_qm9_graphs(dataset, num_graphs):
    for i in range(num_graphs):
        data = dataset[i]
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        G = nx.Graph(edge_list)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.show()


def plot_graph(g, y, label_dict, save=False, show=False):
    name_file = "graph_CORA_basic.png"
    if not os.path.exists(name_file):
        #     plt.figure(figsize=(9, 7))
        #     nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
        # --------
        node_color = []
        nodelist = [[], [], [], [], [], [], []]
        colorlist = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
        ]
        labels = y
        for n, i in enumerate(labels):
            # print("prova", i)
            node_color.append(colorlist[int(i)])
            nodelist[int(i)].append(n)
        pos = nx.spring_layout(g, seed=42)
        plt.figure()
        labellist = list(label_dict.values())
        for num, i in enumerate(zip(nodelist, labellist)):
            n, l = i[0], i[1]
            nx.draw_networkx_nodes(
                g, pos, nodelist=n, node_size=5, node_color=colorlist[num], label=l
            )
        nx.draw_networkx_edges(g, pos, width=0.05, arrowsize=5)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()

        if save:
            plt.savefig(name_file)
        if show:
            plt.show()


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


def create_padded_graph_list(dataset):
    max_num_nodes = max([data.num_nodes for data in dataset])
    graph_list = []
    for data in dataset:
        edge_index = data.edge_index
        edge_list = edge_index.T.tolist()
        G = nx.Graph(edge_list)
        adj = nx.adjacency_matrix(G).todense()
        padded_adj = pad_adjacency_matrix(adj, max_num_nodes)
        graph = {
            "adj": np.array(padded_adj),
            "features": np.array(
                torch.cat(
                    [
                        data.x,
                        torch.zeros(max_num_nodes - data.num_nodes, data.num_features),
                    ]
                )
            ),  # Aggiunta delle features con padding
        }
        graph_list.append(graph)
    return graph_list


def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == "id":
        input_dim = max_num_nodes
    elif args.feature_type == "deg":
        input_dim = 1
    elif args.feature_type == "struct":
        input_dim = 2

    print(input_dim, max_num_nodes)

    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model


def train(args, dataloader, model):
    """
    Trains the model using the provided dataloader and arguments.

    Args:
        args: The arguments for training.
        dataloader: The dataloader for loading the training data.
        model: The model to be trained.

    Returns:
        None
    """
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    for epoch in range(5000):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            features = data["features"].float()
            adj_input = data["adj"].float()

            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()

            loss = model(features, adj_input)
            print("Epoch: ", epoch, ", Iter: ", batch_idx, ", Loss: ", loss)
            loss.backward()

            optimizer.step()
            scheduler.step()
            break


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
        dataset="grid",
        feature_type="id",
        lr=0.001,
        batch_size=1,
        num_workers=1,
        max_num_nodes=-1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA)
    print("CUDA", CUDA)
    ### running log

    if prog_args.dataset == "enzymes":
        graphs = data.Graph_load_batch(min_num_nodes=10, name="ENZYMES")
        num_graphs_raw = len(graphs)
    elif prog_args.dataset == "grid":
        graphs = []
        for i in range(2, 3):
            for j in range(2, 3):
                graphs.append(nx.grid_2d_graph(i, j))
        num_graphs_raw = len(graphs)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    graphs_len = len(graphs)
    print(
        "Number of graphs removed due to upper-limit of number of nodes: ",
        num_graphs_raw - graphs_len,
    )
    graphs_test = graphs[int(0.8 * graphs_len) :]
    # graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_train = graphs

    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(max_num_nodes))

    # dataset = GraphAdjSampler(
    #     graphs_train, max_num_nodes, features=prog_args.feature_type
    # )
    # print("------")
    # print(dataset[0], "type:", (dataset[0]["features"].dtype))
    # print("------")

    # # loading dataset
    dataset = QM9(root="data_prova")
    max_num_nodes = max([data.num_nodes for data in dataset[0:10]])

    dataset = create_padded_graph_list(dataset[0:10])
    print(dataset[0])

    # show_qm9_graphs(dataset, 3)

    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size,
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=prog_args.batch_size, num_workers=prog_args.num_workers
    )
    model = build_model(prog_args, max_num_nodes).cuda()
    train(prog_args, dataset_loader, model)


if __name__ == "__main__":
    main()
