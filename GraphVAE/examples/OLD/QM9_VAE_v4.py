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
from torch.autograd import Variable
import scipy.optimize
from torch import Tensor
from typing import Optional, Tuple, Union
from torch_geometric.utils import (
    train_test_split_edges,
    negative_sampling,
    remove_self_loops,
    add_self_loops,
)

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

    def encode(self, x_features, edge_index):
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

    def decode(self, z, edge_index):
        z = self.conv2(z, edge_index)
        z = F.relu(z)
        z = self.conv3(z, edge_index)
        z = torch.sigmoid(z)
        return z

    def forward(self, x_features, edge_index):
        mu, logvar = self.encode(x_features, edge_index)
        z = self.reparameterize(mu, logvar)

        A_pred = dot_product_decode(z)

        x_pred = self.decode(z, edge_index)
        return A_pred, mu, logvar, z, x_pred

    def generate(self, z):
        A_pred = dot_product_decode(z)
        # z_decoded = self.decode(A_pred, z)
        return A_pred

    def recon_loss(
        self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Optional[Tensor] = None
    ) -> Tensor:

        EPS = 1e-15
        MAX_LOGSTD = 10

        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(self.decode(z, pos_edge_index) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decode(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    # Aggiorna la funzione di loss
    def loss(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        max_num_nodes = 5

        z = self.reparameterize(mu, logvar)
        adj_reconstructed = dot_product_decode(z)
        adj_target = edge_index_to_adjacency(edge_index)
        adj_loss = F.binary_cross_entropy(
            input=adj_reconstructed.view(-1),
            target=adj_target.view(-1).to(dtype=torch.float32),
        )

        pos_loss = -torch.log(self.decode(z, edge_index) + 1e-15).mean()
        # print(self.decode(edge_index, z))

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= max_num_nodes * max_num_nodes  # normalize
        kl_loss = 1 / x.size(0) * kl_divergence

        return pos_loss + kl_loss

    def loss2(self, x, edge_index):
        self.max_num_nodes = 5
        adj_data = edge_index_to_adjacency(edge_index)
        # vae
        z_mu, z_lsgms = self.encode(edge_index, x)

        z = self.reparameterize(mu, logvar)

        out = dot_product_decode(z)

        print(adj_data)
        adj_vectorized = adj_data[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized)

        # print(adj)
        # print('permuted: ', adj_permuted)
        # print('recon: ', recon_adj_tensor)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        print("recon: ", adj_recon_loss)
        print(adj_vectorized_var)
        print(out[0])

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        print("kl: ", loss_kl)

        loss = adj_recon_loss + loss_kl

        return loss

    def edge_similarity_matrix(
        self, adj, adj_recon, matching_features, matching_features_recon, sim_func
    ):
        S = torch.zeros(
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
        )
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = (
                            adj[i, i]
                            * adj_recon[a, a]
                            * sim_func(matching_features[i], matching_features_recon[a])
                        )
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = (
                                adj[i, j]
                                * adj[i, i]
                                * adj[j, j]
                                * adj_recon[a, b]
                                * adj_recon[a, a]
                                * adj_recon[b, b]
                            )
        return S

    def forward_test(self):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor(
            [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        )
        adj_features = torch.Tensor([2, 3, 3, 2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor(
            [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
        )
        adj_features1 = torch.Tensor([3, 3, 2, 2])
        S = self.edge_similarity_matrix(
            adj_data,
            adj_data1,
            adj_features,
            adj_features1,
            self.deg_feature_similarity,
        )

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        # init_assignment = torch.FloatTensor(4, 4)
        # init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        # print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print("row: ", row_ind)
        print("col: ", col_ind)

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print("permuted: ", permuted_adj)

        adj_recon_loss = self.adj_recon_loss(adj_data, adj_data1)
        print(adj_data1)
        print("diff: ", adj_recon_loss)

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [
                        torch.max(x[j, :] * S[i, j, a, :])
                        for j in range(self.max_num_nodes)
                        if j != i
                    ]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x

    def permute_adj(self, adj, curr_ind, target_ind):
        """Permute adjacency matrix.
        The target_ind (connectivity) should be permuted to the curr_ind position.
        """
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int32)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def adj_recon_loss(self, adj_truth, adj_pred):
        print("truth: ", adj_truth)
        return F.binary_cross_entropy(adj_truth, adj_pred)


if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH_SIZE = 1
    LEARNING_RATE = 0.005
    EPOCH = 1000
    latent_space = 8
    hidden_space = 11

    # Caricamento del dataset QM9
    dataset = QM9(root="data/QM9", transform=T.NormalizeFeatures())
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
                data["features"][0], data["edge_index"][0]
            )
            # model.forward_test()
            # mu, logvar = self.encode(x_features, edge_index)
            # z = self.reparameterize(mu, logvar)
            loss = model.loss(data["features"][0], data["edge_index"][0])
            # loss = model.recon_loss(z_star, data["edge_index"][0])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
        print("Epoch: {} Average loss: {:.4f}".format(epoch, loss))

    print("Allenamento completato!")
    A_star, mu, logvar, z_star, x_pred = model(dataset[0].x, dataset[0].edge_index)
    print(" DI NUOVO", x_pred)
    print("MENTRE QUELLO VERO è", dataset[0].x)

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
