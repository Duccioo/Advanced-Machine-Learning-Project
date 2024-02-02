import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GAE, VGAE, GraphConv


from torch_geometric.data import Data

from pyvis.network import Network
import networkx as nx
import torch_geometric.utils as utils
import numpy as np


def custom_collate(batch):
    r"""Custom collate function to handle PyTorch Geometric's Data objects."""
    elem = batch[0]
    if isinstance(elem, Data):
        return batch
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)


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


def main():
    batch_size = 64
    # Caricamento del dataset QM9
    dataset = QM9(root="data/QM9")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )
    # visualizzo il primo grafo:
    data = utils.from_networkx(nx.Graph(edge_index.T.numpy()))
    net = Network(notebook=True)
    net.from_nx(data)
    net.show("graph.html")

    # Inizializzazione del modello e dell'ottimizzatore
    model = GraphVAE(dataset.num_features, 32, dataset.num_features)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Allenamento del modello
    model.train()
    for epoch in range(10):
        train_loss = 0
        for data in loader:
            optimizer.zero_grad()
            data_x = [data[i].x for i in range(len(data))]
            data_edge = [data[i].edge_index for i in range(len(data))]
            print(data_x)
            recon_data, mu, logvar = model(data_x, data_edge)
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


# Funzione per generare nuovi grafi dal modello
def generate_graphs(model, num_graphs):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_graphs, 20)
        adj, features, _ = model.decode(z, None)
        graphs = []
        for i in range(num_graphs):
            edge_index = adj[i].to(torch.long).nonzero().t()
            x = features[i]
            graphs.append((x, edge_index))
        return graphs


if __name__ == "__main__":
    main()
