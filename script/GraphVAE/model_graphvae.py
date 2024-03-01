import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from GraphVAE.model_base import GraphConv, MLP_VAE_plain


# ---
import sys
from os import path

a = sys.path.append((path.dirname(path.dirname(path.abspath(__file__)))))
from utils.utils import graph_to_mol


class GraphVAE(nn.Module):
    def __init__(
        self,
        input_dim,
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

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.num_nodes_features = num_nodes_features
        self.num_edges_features = num_edges_features
        self.input_dimension = input_dim

        self.max_num_edges = max_num_edges
        self.max_num_nodes = max_num_nodes
        self.device = device

        self.vae = MLP_VAE_plain(
            self.input_dimension * self.num_nodes_features,
            latent_dim,
            output_dim,
            device=device,
            e_size=max_num_edges * self.num_edges_features,
        ).to(device=device)

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
                            S[i, i, a, a] = 0

                    # with feature not implemented
                    # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = torch.abs(adj[i, j] - adj_recon[a, b])
        return S

    def mpm(self, x_init, S, max_iters=5):
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

        init_corr = 1 / self.max_num_nodes

        adj_permuted_vectorized = adj_recon_vector.clone().to(self.device)

        # LENTISSIMOO...
        for i in range(adj_recon_vector.shape[0]):
            recon_adj_lower = self.recover_adj_lower(adj_recon_vector[i], self.device)

            adj_wout_diagonal = adj_true[i][
                upper_triangular_indices[0], upper_triangular_indices[1]
            ]
            adj_mask = adj_wout_diagonal.repeat(edges_recon.shape[2], 1).T
            masked_edges_recon_features = edges_recon[i] * adj_mask
            edges_recon_features_total[i] = masked_edges_recon_features.reshape(
                -1, edges_recon.shape[2]
            )

            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

            S = self.edge_similarity_matrix(
                adj_true[i],
                recon_adj_tensor,
                edges_true[i],
                edges_recon[i],
                self.deg_feature_similarity,
            )

            init_assignment = (
                torch.ones(self.max_num_nodes, self.max_num_nodes, device=self.device)
                * init_corr
            )
            assignment = self.mpm(init_assignment, S)

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                -assignment.detach().cpu().numpy()
            )

            adj_permuted = self.permute_adj(adj_true[i], row_ind, col_ind)
            adj_permuted_vectorized[i] = adj_permuted[
                torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
            ]

        adj_recon_loss = F.binary_cross_entropy(
            adj_recon_vector, adj_permuted_vectorized
        )

        loss_kl = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes

        loss_edge = F.mse_loss(edges_recon_features_total, edges_true)
        loss_node = F.mse_loss(node_recon, node_true)

        loss = adj_recon_loss + loss_kl + loss_edge + loss_node

        return loss

    def generate(self, z, smile: bool = False):
        with torch.no_grad():
            # z = z.clone().detach().to(dtype=torch.float32).to(device=device)
            h_decode, output_node_features, output_edge_features = self.vae.decoder(z)

            output_node_features = output_node_features.view(
                -1, self.input_dimension, self.num_nodes_features
            )
            output_edge_features = output_edge_features.view(
                -1, self.max_num_edges, self.num_edges_features
            )
            output_edge_features = F.softmax(output_edge_features, dim=2)
            out = torch.sigmoid(h_decode)

            recon_adj_lower = self.recover_adj_lower(out, device=self.device)
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

            recon_adj_tensor.fill_diagonal_(0)
            recon_adj_tensor = torch.round(recon_adj_tensor)

            n_one = (recon_adj_tensor == 1).sum().item() // 2
            output_edge_features = output_edge_features[:, :n_one]

            # sotto_matrice = torch.round(
            #     F.softmax(output_node_features[:, :, 5:9], dim=2)
            # )

            sotto_matrice = F.softmax(output_node_features[:, :, 5:9], dim=2)
            indici = torch.argmax(sotto_matrice, dim=2)[0]

            mol = None

            if smile:
                mol, smile = graph_to_mol(
                    recon_adj_tensor.cpu(),
                    indici.cpu().numpy(),
                    output_edge_features[0].cpu(),
                    True,
                    True,
                )

            return (
                recon_adj_tensor,
                output_node_features,
                output_edge_features,
                smile,
                mol,
            )

    def save_vae_encoder(self, path):
        self.vae.save_encoder(path)
