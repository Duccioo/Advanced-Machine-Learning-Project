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

    def edge_similarity_matrix(
        self, adj, adj_recon, matching_features, matching_features_recon, sim_func
    ):
        S = torch.zeros(
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
            device=self.device,
        )

        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:

                    for a in range(self.max_num_nodes):

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
                                torch.abs(adj[i, j] - adj_recon[a, b])
                                # * adj[i, i]
                                # * adj[j, j]
                                # * adj_recon[a, b]
                                # * adj_recon[a, a]
                                # * adj_recon[b, b]
                            )
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(
                self.max_num_nodes, self.max_num_nodes, device=self.device
            )
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

        node_recon_features = node_recon_features.view(
            -1, self.max_num_nodes, self.num_nodes_features
        )

        edges_recon_features = edges_recon_features.view(
            -1, self.max_num_edges, self.num_edges_features
        )

        edges_recon_features = F.softmax(edges_recon_features, dim=2)
        out = F.sigmoid(h_decode)

        # recover adj
        adj_permuted_total = torch.empty(out.shape[0], adj.shape[1], adj.shape[2])
        edges_recon_features_total = torch.empty(
            out.shape[0], edges_recon_features.shape[1], edges_recon_features.shape[2]
        )

        upper_triangular_indices = torch.triu_indices(
            row=adj[0].size(0),
            col=adj[0].size(1),
            offset=1,
            device=self.device,
        )

        if out.shape[0] >= 1:
            for i in range(0, out.shape[0]):
                recon_adj_lower = self.recover_adj_lower(out[i], self.device)

                # Otteniamo gli indici della parte triangolare superiore senza la diagonale

                # Estraiamo gli elementi corrispondenti dalla matrice
                adj_wout_diagonal = adj[i][
                    upper_triangular_indices[0], upper_triangular_indices[1]
                ]
                # adj_wout_diagonal = adj[i].triu(diagonal=1).flatten()
                adj_mask = adj_wout_diagonal.repeat(edges_recon_features.shape[2], 1).T

                masked_edges_recon_features = edges_recon_features[i] * adj_mask

                edges_recon_features_total[i] = masked_edges_recon_features.reshape(
                    -1, edges_recon_features.shape[2]
                )

                recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
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
                    torch.ones(
                        self.max_num_nodes, self.max_num_nodes, device=self.device
                    )
                    * init_corr
                )
                # init_assignment = torch.FloatTensor(4, 4)
                # init.uniform(init_assignment)
                assignment = self.mpm(init_assignment, S)
                # matching
                print("qui?")

                # use negative of the assignment score since the alg finds min cost flow
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    -assignment.detach().cpu().numpy()
                )

                # print("row: ", row_ind)
                # print("col: ", col_ind)
                # order row index according to col index

                adj_permuted = self.permute_adj(adj[i], row_ind, col_ind)

                adj_permuted_total[i] = adj_permuted.unsqueeze(0)

                print("allora è questo")

        # print('Assignment: ', assignment)
        print("max num nodes: ", self.max_num_nodes)
        print("max edges nodes: ", self.max_num_edges)

        print(adj.shape)
        print(edges_features.shape)

        print(adj_permuted_total.shape)
        print(edges_recon_features_total.shape)

        adj_vectorized = adj_permuted_total[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.

        adj_recon_loss = F.binary_cross_entropy(out[0], adj_vectorized)

        # print("RECON", recon_adj_tensor)
        # self.generate_features_edge(num_edge = 10)

        # kl loss server solo media e varianza
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        # print("kl: ", loss_kl.item())

        # per quanto riguarda le features degli edge:
        # print(edges_features.shape)
        # print(edges_recon_features.shape)

        loss_edge = F.mse_loss(edges_recon_features_total, edges_features)
        # - MSE tra il target e quello generato stando attenti al numero di archi considerati
        # mascherato con il grafo originale

        # per quanto riguarda le features dei nodi:
        # - MSE di nuovo
        loss_node = F.mse_loss(node_recon_features, nodes_features)

        loss = adj_recon_loss + loss_kl + loss_edge + loss_node

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

        recon_adj_lower = self.recover_adj_lower(out_tensor, device=self.device)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        return recon_adj_tensor, output_node_features, output_edge_features

    def save_vae_encoder(self, path):
        self.vae.save_encoder(path)
