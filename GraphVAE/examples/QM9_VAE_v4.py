import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GAE, VGAE

import networkx as nx
import numpy as np

from torch_geometric.data import Data
from torch.optim.lr_scheduler import MultiStepLR
import torch_geometric.transforms as T

LR_milestones = [500, 1000]


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
            "features": data.x,
            "edge_index": data.edge_index,
        }
        graph_list.append(graph)
    return graph_list, max_num_nodes


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def edge_index_to_adjacency(edge_index):
    """
    Converti una matrice edge_index in una matrice di adiacenza.

    Args:
    - edge_index (numpy.ndarray): Matrice di 2 righe e N colonne contenente gli indici degli archi.

    Returns:
    - torch.Tensor: Matrice di adiacenza.
    """
    # Calcola il numero totale di nodi nel grafo
    num_nodes = max(torch.max(edge_index[0]), torch.max(edge_index[1])) + 1

    # Costruisci la matrice di adiacenza con NumPy e PyTorch
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Assegna 1 agli elementi corrispondenti della matrice di adiacenza
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    return adj_matrix


# Definizione del modello VAE
class GraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)
        self.conv2 = GCNConv(latent_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, in_dim)
        # ---

    def encode(self, edge_index, x_features):
        x = F.relu(self.conv1(x_features, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, edge_index, z):
        z = self.conv2(z, edge_index)
        z = F.relu(z)
        z = self.conv3(z, edge_index)
        z = torch.sigmoid(z)
        return z

    def forward(self, adj, x_features):
        mu, logvar = self.encode(adj, x_features)
        z = self.reparameterize(mu, logvar)

        A_pred = dot_product_decode(z)

        # x_pred = self.decode(adj, z)
        return A_pred, mu, logvar, z, 0

    def generate(self, z):
        A_pred = dot_product_decode(z)
        return A_pred

    # Aggiorna la funzione di loss
    def loss(self, x, edge_index):
        mu, logvar = self.encode(edge_index, x)

        z = self.reparameterize(mu, logvar)
        adj_reconstructed = dot_product_decode(z)
        adj_target = edge_index_to_adjacency(edge_index)
        # adj_loss = F.binary_cross_entropy(
        #     input=adj_reconstructed.view(-1),
        #     target=adj_target.view(-1).to(dtype=torch.float32),
        # )

        pos_loss = -torch.log(self.decode(edge_index, z) + 1e-15).mean()
        # print(self.decode(edge_index, z))

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # kl_divergence /= max_num_nodes * max_num_nodes  # normalize
        kl_loss = 1 / x.size(0) * kl_divergence

        return pos_loss + kl_loss


if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH_SIZE = 1
    LEARNING_RATE = 0.005
    EPOCH = 3000
    latent_space = 8
    hidden_space = 11

    # Caricamento del dataset QM9
    dataset = QM9(
        root="data/QM9",
    )
    dataset_padded, max_number_nodes = create_padded_graph_list(dataset[0:1])

    loader = DataLoader(dataset_padded, batch_size=BATCH_SIZE, shuffle=False)

    all_adj = np.empty(
        (
            len(dataset_padded),
            dataset_padded[0]["adj"].shape[0],
            dataset_padded[0]["adj"].shape[1],
        )
    )
    for adj in dataset_padded:
        all_adj = np.append(all_adj, adj["adj"])

    norm = (
        all_adj.shape[0]
        * all_adj.shape[0]
        / float((all_adj.shape[0] * all_adj.shape[0] - all_adj.sum()) * 2)
    )

    # Inizializzazione del modello e dell'ottimizzatore
    print("Istanzio il modello")
    model = GraphVAE(
        dataset.num_features, hidden_dim=hidden_space, latent_dim=latent_space
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=LEARNING_RATE)

    print("\nTRAINING\n")
    # Allenamento del modello
    model.train()
    for epoch in range(EPOCH):
        train_loss = 0
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()

            A_star, mu, logvar, z_star, x_pred = model(
                data["edge_index"][0], data["features"][0]
            )
            loss = model.loss(data["features"][0], data["edge_index"][0])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
        print("Epoch: {} Average loss: {:.4f}".format(epoch, loss))

    print("Allenamento completato!")
    A_star, mu, logvar, z_star, x_pred = model(dataset[0].edge_index, dataset[0].x)
    print(" DI NUOVO A_STAR", A_star)

    # Generazione di un vettore di rumore casuale
    z = torch.randn(5, latent_space)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj = model.generate(z)

    rounded_matrix = torch.round(adj)
    print(dataset[0])
    print(dataset_padded[0]["features"])
    print(" true edge_index", dataset_padded[0]["edge_index"])
    print(" true adj", dataset_padded[0]["adj"])
    print(rounded_matrix)
