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
        max_num_edges,
        pool="sum",
        num_nodes_features=11,
        num_edges_features=4,
        device=torch.device("cpu"),
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
        self.num_nodes_features = num_nodes_features
        self.num_edges_features = num_edges_features
        self.input_dimension = input_dim

        self.max_num_edges = max_num_edges
        self.max_num_nodes = max_num_nodes
        self.device = device

        # self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.vae = MLP_VAE_plain(
            self.input_dimension * self.num_nodes_features,
            latent_dim,
            output_dim,
            device=device,
            e_size=max_num_edges * self.num_edges_features,
        )

        # self.feature_mlp = MLP_plain(latent_dim, latent_dim, output_dim)

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
        ind = np.zeros(self.max_num_nodes, dtype=np.int32)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros(
            (self.max_num_nodes, self.max_num_nodes), device=self.device
        )
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == "max":
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == "sum":
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def deg_feature_similarity(self, f1, f2):
        result = 1 / (abs(f1 - f2) + 1)
        # print(result.shape)
        return result

    def deg_feature_similarity_2(self, f1, f2):
        edge_similarity = F.cosine_similarity(f1, f2, dim=0)
        return edge_similarity

    def edge_similarity_matrix_2(self, adj_1, adj_2, features_1, features_2):
        # Calcola la similarità tra features degli edge utilizzando il coseno
        edge_similarity = F.cosine_similarity(features_1, features_2, dim=1)

        # Calcola la similarità tra matrici adiacenti utilizzando il coseno
        adj_similarity = F.cosine_similarity(adj_1.view(1, -1), adj_2.view(1, -1))

        # print("adj_similarity", adj_similarity.shape)

        # Calcola la matrice di similarità degli edge
        edge_similarity_matrix = edge_similarity * adj_similarity
        return edge_similarity_matrix

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

                        # print(i, a)
                        # print(adj[i, i])
                        # print(adj_recon[a, a])
                        # calcolo la similarità nei loop
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
                                # * adj[i, i]
                                # * adj[j, j]
                                * adj_recon[a, b]
                                # * adj_recon[a, a]
                                # * adj_recon[b, b]
                            )
        return S

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

    def forward(self, adj, edges_features, nodes_features):
        # x = self.conv1(input_features, adj)
        # x = self.bn1(x)
        # x = self.act(x)
        # x = self.conv2(x, adj)
        # x = self.bn2(x)

        # pool over all nodes
        # graph_h = self.pool_graph(x)
        graph_h = nodes_features.reshape(
            -1, self.input_dimension * self.num_nodes_features
        )

        # vae
        h_decode, z_mu, z_lsgms, node_recon_features, edges_recon_features = self.vae(
            graph_h
        )

        edges_recon_features = edges_recon_features.view(
            -1, self.max_num_edges, self.num_edges_features
        )

        edges_recon_features = F.softmax(edges_recon_features, dim=2)
        # print(edges_recon_features)
        
        out = F.sigmoid(h_decode)

        # recover adj
        adj_permuted_total = adj[0].reshape(1, adj.shape[1], adj.shape[2])
        if out.shape[0] > 1:
            for i in range(0, out.shape[0]):
                recon_adj_lower = self.recover_adj_lower(out[i], self.device)
                recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

                # print(adj[i].shape)
                # print(recon_adj_tensor.shape)
                # print(edges_features[i].shape)
                # print(edges_recon_features[i].shape)

                S = self.edge_similarity_matrix(
                    adj[i],
                    recon_adj_tensor,
                    edges_features[i],
                    edges_recon_features[i],
                    self.deg_feature_similarity_2,
                )

                # initialization strategies
                init_corr = 1 / self.max_num_nodes
                init_assignment = (
                    torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
                )
                # init_assignment = torch.FloatTensor(4, 4)
                # init.uniform(init_assignment)
                assignment = self.mpm(init_assignment, S)
                # matching
                # use negative of the assignment score since the alg finds min cost flow
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    -assignment.detach().numpy()
                )
                # print("row: ", row_ind)
                # print("col: ", col_ind)
                # order row index according to col index

                adj_data = adj[i]
                adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)

                adj_permuted_total = torch.cat(
                    (adj_permuted_total, adj_permuted.unsqueeze(0)), dim=0
                )

        # print('Assignment: ', assignment)

        adj_vectorized = adj_permuted[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.
        # end_adj_true_vectorization = time.time() - start_adj_true_vectorization

        adj_recon_loss = F.binary_cross_entropy(out[0], adj_vectorized)

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
        h_decode, output_node_features, output_edge_features = self.vae.decode(z)
        output_node_features = output_node_features.view(
            -1, self.input_dimension, self.num_nodes_features
        )
        # print("output features ", output_features.shape)
        output_node_features = output_node_features[
            :, :, : self.num_nodes_features
        ].squeeze_()
        # Definizione dei bin per la classificazione
        num_bins = 5
        bins = torch.linspace(0, 1, num_bins + 1).to(device)  # 5 bin per 5 classi
        # Classificazione dei valori del vettore in base ai bin
        # classifications = np.digitize(
        #     output_node_features[:, 5:6].detach().to(device="cpu"), bins
        # )
        classifications = torch.bucketize(output_node_features[:, 5:6], bins)

        # output_node_features[:, 5] = classifications.squeeze_()
        
        print(output_edge_features.shape)
        
        output_edge_features = output_edge_features.view(
            -1, self.max_num_edges, self.num_edges_features
        )

        output_edge_features = F.softmax(output_edge_features, dim=2)
        print(output_edge_features.shape)
        
        # Crea una matrice one-hot utilizzando max_indices
        # print("output nodes ", output_node_features.shape)

        # print("output edges ", output_edge_features.shape)

        out = F.sigmoid(h_decode)
        out_tensor = out.data

        # print(out_tensor.shape)

        # print("forward completato")
        recon_adj_lower = self.recover_adj_lower(out_tensor, device=self.device)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
        
        
        # print("adj ripresa")
        # print(recon_adj_tensor)

        return recon_adj_tensor, output_node_features, output_edge_features



