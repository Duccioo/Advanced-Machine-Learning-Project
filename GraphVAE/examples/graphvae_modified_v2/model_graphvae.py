import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from model import GraphConv, MLP_VAE_plain

import time


class GraphVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        max_num_nodes,
        pool="sum",
        num_features=11,
    ):
        """
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        """
        super(GraphVAE, self).__init__()
        # self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        # self.bn1 = nn.BatchNorm1d(input_dim)
        # self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        # self.bn2 = nn.BatchNorm1d(input_dim)
        # self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.num_features = num_features
        self.input_dimension = input_dim
        # self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.vae = MLP_VAE_plain(
            self.input_dimension * self.num_features,
            latent_dim,
            output_dim,
            device=torch.device("cuda"),
        )

        # self.feature_mlp = MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l, device=torch.device("cuda")):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes, device=device)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def permute_adj(self, adj, curr_ind, target_ind):
        """Permute adjacency matrix.
        The target_ind (connectivity) should be permuted to the curr_ind position.
        """
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == "max":
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == "sum":
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj):
        # x = self.conv1(input_features, adj)
        # x = self.bn1(x)
        # x = self.act(x)
        # x = self.conv2(x, adj)
        # x = self.bn2(x)

        # pool over all nodes
        # graph_h = self.pool_graph(x)
        graph_h = input_features.reshape(-1, self.input_dimension * self.num_features)

        # vae
        h_decode, z_mu, z_lsgms, output_features = self.vae(graph_h)
        out = F.sigmoid(h_decode)

        adj_data = adj.data[0]
        adj_vectorized = adj_data[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.
        # end_adj_true_vectorization = time.time() - start_adj_true_vectorization

        adj_recon_loss = F.binary_cross_entropy(out[0], adj_vectorized)

        # recon_adj_lower = self.recover_adj_lower(out)
        # recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # print("RECON", recon_adj_tensor)
        # self.generate_features_edge(num_edge = 10)

        # kl loss server solo media e varianza
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        # print("kl: ", loss_kl.item())

        loss = adj_recon_loss + loss_kl

        # end_forward = time.time() - start_forward
        # print("FORWARD: ", end_vae, end_adj_true_vectorization, end_forward)

        return loss

    def loss(self, adj_true, adj_recon, mu, var):

        adj_vectorized_true = adj_true[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.

        adj_vectorized_recon = adj_recon[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.

        # F.binary_cross_entropy(adj_vectorized_recon, adj_vectorized_true)
        adj_recon_loss = F.binary_cross_entropy(
            adj_vectorized_recon, adj_vectorized_true
        )
        # self.generate_features_edge(num_edge = 10)

        # kl loss server solo media e varianza
        loss_kl = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        # print("kl: ", loss_kl.item())

        loss = adj_recon_loss + loss_kl

        return loss

    def generate(self, z, device="cpu"):
        # print("z shape", z.shape)
        # Variable(torch.tensor(input_features, dtype=torch.float32))
        # z = torch.tensor(z, dtype=torch.float32).cuda()
        z = z.clone().detach().to(dtype=torch.float32).to(device=device)
        print("z shape", z.shape)
        # input_features = np.array(input_features)
        # print(input_features)
        # graph_h = input_features.view(-1, self.max_num_nodes * self.num_features)
        # h_decode, z_mu, z_lsgms = self.vae(graph_h)
        h_decode, output_features = self.vae.decode(z)
        output_features = output_features.view(
            -1, self.input_dimension, self.num_features
        )
        # print("output features ", output_features.shape)
        output_node_features = output_features[:, :, : self.num_features].squeeze_()
        # Definizione dei bin per la classificazione
        num_bins = 5
        bins = torch.linspace(0, 1, num_bins + 1).to(device)  # 5 bin per 5 classi
        # Classificazione dei valori del vettore in base ai bin
        # classifications = np.digitize(
        #     output_node_features[:, 5:6].detach().to(device="cpu"), bins
        # )
        classifications = torch.bucketize(output_node_features[:, 5:6], bins)

        # output_node_features[:, 5] = classifications.squeeze_()

        output_edge_features = output_features[:, :, self.num_features - 4 :].squeeze_()
        max_indices = torch.argmax(output_edge_features, dim=1)
        # Crea una matrice one-hot utilizzando max_indices
        output_edge_features = torch.eye(4)[max_indices.detach().to(device="cpu")]
        # print("output nodes ", output_node_features.shape)
        # print("output edges ", output_edge_features.shape)

        out = F.sigmoid(h_decode)
        out_tensor = out.data

        # print(out_tensor.shape)

        # print("forward completato")
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
        # print("adj ripresa")
        # print(recon_adj_tensor)

        return recon_adj_tensor, output_node_features, output_edge_features
