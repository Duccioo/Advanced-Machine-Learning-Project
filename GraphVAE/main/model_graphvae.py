import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from model import GraphConv, MLP_VAE_plain

import time

# ---
from data_graphvae import data_to_smiles, graph_to_mol


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
                        try:
                            S[i, i, a, a] = (
                                adj[i, i]
                                * adj_recon[a, a]
                                * sim_func(
                                    matching_features[i], matching_features_recon[a]
                                )
                            )
                        except:
                            matching_features = matching_features.repeat(2, 1)
                            matching_features_recon = matching_features_recon.repeat(
                                2, 1
                            )
                            S[i, i, a, a] = (
                                adj[i, i]
                                * adj_recon[a, a]
                                * sim_func(
                                    matching_features[i], matching_features_recon[a]
                                )
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

    def forward(self, nodes_features):
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

        return out, z_mu, z_lsgms, node_recon_features, edges_recon_features

    def loss(
        self,
        adj_true,
        adj_recon_vector,
        node_true,
        node_recon,
        edges_true,
        edges_recon,
        mu,
        var,
    ):
        # recover adj
        edges_recon_features_total = torch.empty(
            adj_true.shape[0],
            edges_recon.shape[1],
            edges_recon.shape[2],
            device=self.device,
        )

        upper_triangular_indices = torch.triu_indices(
            row=adj_true[0].size(0),
            col=adj_true[0].size(1),
            offset=1,
            device=self.device,
        )

        adj_permuted_vectorized = torch.empty(
            adj_recon_vector.size(0), adj_recon_vector.size(1), device=self.device
        )

        if adj_recon_vector.shape[0] >= 1:
            for i in range(0, adj_recon_vector.shape[0]):
                recon_adj_lower = self.recover_adj_lower(
                    adj_recon_vector[i], self.device
                )

                # Otteniamo gli indici della parte triangolare superiore senza la diagonale

                # Estraiamo gli elementi corrispondenti dalla matrice
                adj_wout_diagonal = adj_true[i][
                    upper_triangular_indices[0], upper_triangular_indices[1]
                ]
                # adj_wout_diagonal = adj[i].triu(diagonal=1).flatten()
                adj_mask = adj_wout_diagonal.repeat(edges_recon.shape[2], 1).T

                masked_edges_recon_features = edges_recon[i] * adj_mask

                edges_recon_features_total[i] = masked_edges_recon_features.reshape(
                    -1, edges_recon.shape[2]
                )

                recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

                # print(adj[i].shape)
                # print(recon_adj_tensor.shape)
                # print(edges_features[i].shape)
                # print(edges_recon_features[i].shape)

                S = self.edge_similarity_matrix(
                    adj_true[i],
                    recon_adj_tensor,
                    edges_true[i],
                    edges_recon[i],
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

                # use negative of the assignment score since the alg finds min cost flow
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    -assignment.detach().cpu().numpy()
                )

                # print("row: ", row_ind)
                # print("col: ", col_ind)
                # order row index according to col index

                adj_permuted = self.permute_adj(adj_true[i], row_ind, col_ind)
                # adj_permuted_total[i] = adj_permuted.unsqueeze(0)
                adj_permuted_vectorized[i] = adj_permuted[
                    torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
                ]  # qui si va a trasformare la matrice adiacente in un vettore prendento la triangolare superiore della matrice adiacente.

        adj_recon_loss = F.binary_cross_entropy(
            adj_recon_vector, adj_permuted_vectorized
        )

        # kl loss server solo media e varianza
        loss_kl = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize

        # per quanto riguarda le features degli edge:

        loss_edge = F.mse_loss(edges_recon_features_total, edges_true)
        # - MSE tra il target e quello generato stando attenti al numero di archi considerati
        # mascherato con il grafo originale

        # per quanto riguarda le features dei nodi:
        # - MSE di nuovo
        loss_node = F.mse_loss(node_recon, node_true)

        loss = adj_recon_loss + loss_kl + loss_edge + loss_node

        # end_forward = time.time() - start_forward
        # print("FORWARD: ", end_vae, end_adj_true_vectorization, end_forward)

        return loss

    def generate(self, z, device="cpu", smile=False):
        # z = torch.tensor(z, dtype=torch.float32).cuda()
        z = z.clone().detach().to(dtype=torch.float32).to(device=device)
        # input_features = np.array(input_features)
        # print(input_features)
        # graph_h = input_features.view(-1, self.max_num_nodes * self.num_features)
        # h_decode, z_mu, z_lsgms = self.vae(graph_h)
        with torch.no_grad():
            h_decode, output_node_features, output_edge_features = self.vae.decoder(z)
            output_node_features = output_node_features.view(
                -1, self.input_dimension, self.num_nodes_features
            )
            # print("output features ", output_features.shape)

            output_edge_features = output_edge_features.view(
                -1, self.max_num_edges, self.num_edges_features
            )

            output_edge_features = F.softmax(output_edge_features, dim=2)
            out = F.sigmoid(h_decode)
            # out_tensor = out.data

            recon_adj_lower = self.recover_adj_lower(out, device=self.device)
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

            # Crea una maschera booleana per gli elementi uguali a 1 sulla diagonale
            maschera_diagonale = torch.eye(recon_adj_tensor.shape[0], dtype=bool)

            # Imposta gli elementi sulla diagonale a 0
            recon_adj_tensor[maschera_diagonale] = 0
            recon_adj_tensor = torch.round(recon_adj_tensor)

            # Crea una maschera booleana per gli elementi uguali a 1
            maschera_1 = recon_adj_tensor == 1

            # Conta il numero di 1 nella matrice
            n_one = maschera_1.sum().item() // 2

            output_edge_features = output_edge_features[:, :n_one]

            # riconverto il one-hot encoding in numero atomico:
            # Prendi le colonne dal 5 al 10
            sotto_matrice = F.softmax(output_node_features[:, :, 5:9], dim=2)

            # Trova l'indice del valore massimo lungo l'asse delle colonne
            indici = torch.argmax(sotto_matrice, dim=2)[0]

            if smile:
                smile = graph_to_mol(
                    recon_adj_tensor.to("cpu"),
                    indici.to("cpu").numpy(),
                    output_edge_features[0].to("cpu"),
                    True,
                    True,
                )

            return recon_adj_tensor, output_node_features, output_edge_features, smile

    def save_vae_encoder(self, path):
        self.vae.save_encoder(path)
