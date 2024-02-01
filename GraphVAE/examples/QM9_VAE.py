import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GAE, VGAE

from pyvis.network import Network
import networkx as nx
import torch_geometric.utils as utils
import numpy as np


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


# Definizione del modello VAE
class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphVAE, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, edge_index):
        return torch.sigmoid(self.conv1(z, edge_index))

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar


if __name__ == "__main__":
    # Caricamento del dataset QM9
    dataset = QM9(root="data/QM9")
    # dataset_padded = create_padded_graph_list(dataset[0:10])

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(dataset[0])
    
    # dataset = QM9(root="data/QM9")
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Inizializzazione del modello e dell'ottimizzatore
    model = GraphVAE(dataset.num_features, 32, dataset.num_features)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Allenamento del modello
    model.train()
    for epoch in range(10):
        train_loss = 0
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()
            print(data)
            recon_data, mu, logvar = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(recon_data, data.x, reduction="sum")
            loss += (
                (1 / data.num_nodes)
                * 0.5
                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            "Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(loader.dataset)
            )
        )

    print("Allenamento completato!")

    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, 20)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj, features, _ = model.decode(z, None)
        edge_index = adj[0].to(torch.long).nonzero().t()
        x = features[0]

    # Visualizzazione del grafo generato
    data = utils.from_networkx(nx.Graph(edge_index.T.numpy()))
    net = Network(notebook=True)
    net.from_nx(data)
    net.show("graph.html")
