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

from torch_geometric.data import Data
from torch.nn import MSELoss, BCELoss
from torch.optim.lr_scheduler import MultiStepLR
import torch_geometric.transforms as T

LR_milestones = [500, 1000]


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
            "features": data.x
            # np.array(
            # torch.cat(
            #     [
            #         data.x,
            #         torch.zeros(max_num_nodes - data.num_nodes, data.num_features),
            #     ]
            # )
            ,  # Aggiunta delle features con padding
            "edge_index": data.edge_index,
        }
        graph_list.append(graph)
    return graph_list, max_num_nodes


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


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
        # print("edge_index e x_features", edge_index.shape, x_features.shape)
        # print("x_features", x_features)
        # print("edge_index", edge_index)
        x = F.relu(self.conv1(x_features, edge_index))
        # x = F.relu(self.linear1(x_features))
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
        # print("z entrato nel decode", z.shape, edge_index.shape)
        # print(self.conv2)
        z = self.conv2(z, edge_index)
        z = F.relu(z)
        z = self.conv3(z, edge_index)
        # z = F.relu(self.linear_decode1(z))
        # z = self.linear_decode2(z)
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
        # print("A_pred", A_pred.shape)
        return A_pred


# Aggiorna la funzione di loss
def loss_function(
    adj_reconstructed,
    adj_target,
    feat_reconstructed,
    feat_target,
    mean,
    logvar,
    max_num_nodes,
):
    # Calcola la BCELoss per la matrice di adiacenza
    adj_loss = F.binary_cross_entropy(
        input=adj_reconstructed.view(-1),
        target=adj_target.view(-1).to(dtype=torch.float32),
    )


    # Calcola la MSELoss per le features
    # feat_loss = MSELoss()(feat_reconstructed, feat_target)

    # Calcola la divergenza KL tra la distribuzione latente e la distribuzione normale
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_divergence /= max_num_nodes * max_num_nodes  # normalize

    # Combina i termini di loss
    total_loss = adj_loss + kl_divergence  # + feat_loss

    return total_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    # Caricamento del dataset QM9
    dataset = QM9(root="data/QM9", transform=T.NormalizeFeatures())
    dataset_padded, max_number_nodes = create_padded_graph_list(dataset[0:1])
    BATCH_SIZE = 1
    LEARNING_RATE = 0.01
    # print("dataset_padded", dataset_padded[0:4])
    # dataset_padded = create_padded_graph_list(dataset[0:10])

    # loader = DataLoader(
    #     dataset[0:1000], batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
    # )
    loader = DataLoader(dataset_padded, batch_size=BATCH_SIZE, shuffle=False)
    


    latent_space = 2
    hidden_space = 5

    print(dataset_padded[0])
    print(dataset_padded[0]["adj"].shape)

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

    # dataset = QM9(root="data/QM9")
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Inizializzazione del modello e dell'ottimizzatore
    print(dataset.num_features)

    print("Istanzio il modello")
    model = GraphVAE(
        dataset.num_features, hidden_dim=hidden_space, latent_dim=latent_space
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=LEARNING_RATE)

    EPOCH = 2000

    print("\nTRAINING\n")
    # Allenamento del modello
    model.train()
    A_star, mu, logvar, z_star, x_pred = model(dataset[0].edge_index, dataset[0].x)
    print(A_star)
    for epoch in range(EPOCH):
        train_loss = 0
        for batch_idx, data in enumerate(loader):
            # print(data)
            optimizer.zero_grad()
            # print("adj", data["adj"].shape)
            # print("edge index", data["edge_index"].shape)
            # print("features", data["features"].shape)

            # print("-" * 10)
            A_star, mu, logvar, z_star, x_pred = model(
                data["edge_index"][0], data["features"][0]
            )
            # print("-" * 10)

            # recon_x, kl_loss = model(data[0])

            # input_x = [elem.x for elem in data]
            # print(input_x[0])
            # recon_data, mu, logvar = model(
            #     np.array(input_x),
            #     ([elem.edge_index for elem in data]),
            # )
            # print(recon_data.shape)
            # # print(data[0].x)
            loss = loss_function(
                A_star,
                data["adj"][0],
                x_pred,
                data["features"][0],
                mu,
                logvar,
                max_num_nodes=2,
            )
            # loss = loss_function_old(
            #     A_star, data["adj"][0], mu, logvar, normalization=norm
            # )
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

    # # Visualizzazione del grafo generato
    # data = utils.from_networkx(nx.Graph(adj))
    # net = Network(notebook=True)
    # net.from_nx(data)
    # net.show("graph.html")
