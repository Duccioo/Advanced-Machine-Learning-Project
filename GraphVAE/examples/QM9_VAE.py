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
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)
        self.linear_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.linear_decode2 = nn.Linear(hidden_dim, in_dim)
        # self.vgae = VGAE(hidden_dim, latent_dim)
        # self.encoder = GCNConv(in_dim, hidden_dim)

    # def encode(self, x, edge_index):
    #     # print("x e edge_index", x.shape, edge_index.shape)
    #     x = F.relu(self.conv1(x, edge_index))
    #     print("x 2", x.shape)
    #     return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    def encode(self, edge_index, x_features):
        # print("edge_index e x_features", edge_index.shape, x_features.shape)
        # x = F.relu(self.conv1(x_features, edge_index))
        x = F.relu(self.linear1(x_features))
        # print("x", x.shape)
        # mu = self.conv_mu(x, edge_index)
        mu = self.linear_mu(x)
        # print("mu", mu.shape)
        # logvar = self.conv_logvar(x, edge_index)
        logvar = self.linear_logvar(x)
        # print("logvar", logvar.shape)
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
        # z = self.conv2(z, edge_index)
        # z = F.relu(z)
        # z = self.conv3(z, edge_index)
        z = F.relu(self.linear_decode1(z))
        z = F.relu(self.linear_decode2(z))
        # z = torch.sigmoid(z)
        return z

    def forward(self, adj, x_features):
        mu, logvar = self.encode(adj, x_features)
        # print("mu e logvar", mu, logvar)
        # print("mu e logvar", mu.shape, logvar.shape)
        z = self.reparameterize(mu, logvar)

        # print("z uscita dall'encode e parametrizzata", z.shape)
        # z_decoded = self.decode(z, edge_index)
        # print("z_decoded", z_decoded.shape)
        # return z_decoded, mu, logvar

        A_pred = dot_product_decode(z)
        # print("z", z.shape)

        x_pred = self.decode(adj, z)
        # print(" x_pred", x_pred.shape)
        # print("A_pred", A_pred.shape)
        return A_pred, mu, logvar, z, x_pred

    def generate(self, z):
        A_pred = dot_product_decode(z)
        # print("A_pred", A_pred.shape)
        return A_pred


def loss_function_old(A_pred, A_true, mu, logvar, normalization):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # MSE = F.mse_loss(recon_x, x)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Create Model
    # pos_weight = float(A_pred.shape[0] * A_pred.shape[0] - A_pred.sum()) / A_pred.sum()

    # weight_mask = A_true.to_dense().view(-1) == 1
    # weight_tensor = torch.ones(weight_mask.size(0))
    # weight_tensor[weight_mask] = pos_weight

    loss = normalization * F.binary_cross_entropy(
        A_pred.view(-1),
        A_true.to_dense().view(-1).to(dtype=torch.float32),  # weight=weight_tensor
    )

    kl_divergence = (
        0.5
        / A_pred.size(0)
        * (1 + 2 * logvar - mu**2 - torch.exp(logvar) ** 2).sum(1).mean()
    )
    loss -= kl_divergence
    return loss


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
    # adj_loss = BCELoss()(
    #     adj_reconstructed.view(-1), adj_target.view(-1).to(dtype=torch.float32)
    # )
    adj_loss = F.binary_cross_entropy(
        input=adj_reconstructed.view(-1),
        target=adj_target.view(-1).to(dtype=torch.float32),
    )

    # Calcola la MSELoss per le features
    feat_loss = MSELoss()(feat_reconstructed, feat_target)

    # Calcola la divergenza KL tra la distribuzione latente e la distribuzione normale
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_divergence /= max_num_nodes * max_num_nodes  # normalize

    # Combina i termini di loss
    total_loss = adj_loss + kl_divergence  # + feat_loss

    return total_loss


if __name__ == "__main__":
    # Caricamento del dataset QM9
    dataset = QM9(root="data/QM9")
    dataset_padded, max_number_nodes = create_padded_graph_list(dataset[0:1])
    BATCH_SIZE = 1
    LEARNING_RATE = 0.01
    # print("dataset_padded", dataset_padded[0:4])
    # dataset_padded = create_padded_graph_list(dataset[0:10])

    # loader = DataLoader(
    #     dataset[0:1000], batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
    # )
    loader = DataLoader(dataset_padded, batch_size=BATCH_SIZE, shuffle=False)

    latent_space = 5
    hidden_space = 8

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

    EPOCH = 5000

    print("\nTRAINING\n")
    # Allenamento del modello
    model.train()
    for epoch in range(EPOCH):
        train_loss = 0
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()
            # print("adj", data["adj"].shape)
            # print("edge index", data["edge_index"].shape)
            # print("features", data["features"].shape)

            # print("-" * 10)
            data_prova_edge_index = torch.tensor([[0., 0.], [0., 0.]], dtype=torch.float32)
            data_prova_features = torch.tensor(
                [
                    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
            )
            data_prova_adj = torch.tensor([[0., 0.], [0., 0.]], dtype=torch.float32)
            A_star, mu, logvar, z_star, x_pred = model(
                data_prova_edge_index, data_prova_features
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
                data_prova_adj,
                x_pred,
                data_prova_features,
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
        print(
            "Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(loader.dataset)
            )
        )

    print("Allenamento completato!")

    # Generazione di un vettore di rumore casuale
    z = torch.randn(2, latent_space)

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
