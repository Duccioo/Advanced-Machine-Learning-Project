import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ---
from GraphVAE.model_base import (
    GraphConv,
    MLP_VAE_plain_DECODER,
    MLP_VAE_plain_ENCODER,
)

import sys
from os import path

a = sys.path.append((path.dirname(path.dirname(path.abspath(__file__)))))


def hungarian_algorithm(costs):
    # print(costs)
    # obtain a column vector of minimum row values
    row_mins, _ = torch.min(costs, dim=1, keepdim=True)
    # subtract the tensor of minimum values (broadcasting the minimum value over each row)
    costs = costs - row_mins
    # obtain a row vector of minimum column values
    col_mins, _ = torch.min(costs, dim=0, keepdim=True)
    # subtract the tensor of minimum values (broadcasting the minimum value over each column)
    costs = costs - col_mins
    # proceed with partial assignment
    row_zero_counts = costs.size(1) - torch.count_nonzero(costs, dim=1)
    assigned_columns = []
    assigned_rows = []
    assignment = []
    # assign rows in progressive order of available options
    for opt in range(1, torch.max(row_zero_counts) + 1):
        for i in torch.argwhere(row_zero_counts == opt):
            for j in torch.argwhere(costs[i, :] == 0)[:, 1:]:
                if i.item() not in assigned_rows and j.item() not in assigned_columns:
                    assigned_rows.append(i.item())
                    assigned_columns.append(j.item())
                    assignment.append(torch.concatenate((i, j)))
    # refine assignment until all the rows and columns are assigned
    while len(assignment) < costs.size(0):
        # mark unassigned rows
        marked_rows = list(set(range(costs.size(0))) - set(assigned_rows))
        # build queue of rows to examine
        row_queue = list(set(range(costs.size(0))) - set(assigned_rows))
        # initialize empty list of marked columns
        marked_columns = []
        # initialize empty queue of columns to examine
        column_queue = []
        # examine rows and columns until everything marked is examined
        while len(row_queue) > 0:
            # mark columns with zeros in marked rows
            for j in torch.argwhere(costs[row_queue, :] == 0):
                if j[1].item() not in marked_columns:
                    marked_columns.append(j[1].item())
                    column_queue.append(j[1].item())
            # empty row queue
            row_queue = []
            # mark assigned rows with assignment on marked columns in the queue
            for t in assignment:
                if t[1].item() in column_queue and t[0].item() not in marked_rows:
                    marked_rows.append(t[0].item())
                    row_queue.append(t[0].item())
            # empty column queue
            column_queue = []
        # obtain minimum uncovered element (on marked rows and unmarked columns)

        try:

            min_value = torch.min(
                costs[marked_rows, :][:, list(set(range(costs.size(1))) - set(marked_columns))]
            )
        except:
            min_value = 0

        # subtract minimum value from uncovered elements
        # print(min_value.item())
        # print(costs[0, 0])

        for i in marked_rows:
            for j in list(set(range(costs.size(1))) - set(marked_columns)):
                costs[i, j] = costs[i, j] - min_value
        # and minimum value to double-covered elements
        for i in list(set(range(costs.size(0))) - set(marked_rows)):
            for j in marked_columns:
                costs[i, j] = costs[i, j] + min_value
        # re-assign everything
        row_zero_counts = costs.size(1) - torch.count_nonzero(costs, dim=1)
        assigned_columns = []
        assigned_rows = []
        assignment = []
        # assign rows in progressive order of available options
        for opt in range(1, torch.max(row_zero_counts) + 1):
            for i in torch.argwhere(row_zero_counts == opt):
                for j in torch.argwhere(costs[i, :] == 0)[:, 1:]:
                    if i.item() not in assigned_rows and j.item() not in assigned_columns:
                        assigned_rows.append(i.item())
                        assigned_columns.append(j.item())
                        assignment.append(torch.concatenate((i, j)))
    # obtain final assignment tensor
    assignment = [torch.reshape(a, (1, 2)) for a in assignment]
    assignment = torch.concatenate(assignment, dim=0)
    # return row indices and column inices as separate tensors
    print("MA QUI CI SONO?")
    return assignment[:, 0], assignment[:, 1]


class GraphVAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        max_num_nodes,
        max_num_edges,
        num_nodes_features=11,
        num_edges_features=4,
        device=torch.device("cpu"),
        pool="sum",
    ):

        super(GraphVAE, self).__init__()

        self.device = device

        self.num_nodes_features = num_nodes_features
        self.num_edges_features = num_edges_features

        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

        self.h_size = (
            self.max_num_nodes * self.num_nodes_features
        )  # dimensione dell'input, in questo caso schiaccio la matrice delle features
        self.embedding_size = latent_dim
        self.e_size = (
            self.max_num_edges * self.num_edges_features
        )  # dimensione dell'output della matrice delle features degli edges
        self.output_dim = max_num_nodes * (max_num_nodes + 1) // 2

        # definizione dei componenti della VAE:
        self.encoder = MLP_VAE_plain_ENCODER(self.h_size, self.embedding_size, device).to(device)

        self.decoder = MLP_VAE_plain_DECODER(
            self.h_size, self.embedding_size, self.output_dim, self.e_size
        ).to(device)

        # inizializzo i pesi con xavier
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, vector):
        """
        Recover adjacency upper triangular matrix from vector
        """
        rows, _ = vector.size()

        # Creare una matrice con gli zeri
        adj = torch.zeros(rows, self.max_num_nodes, self.max_num_nodes, device=self.device)
        adj[torch.triu(torch.ones(rows, self.max_num_nodes, self.max_num_nodes)) == 1] = vector.view(1, -1)

        return adj

    def recover_full_adj_from_lower(self, lower):
        """
        Recover the full adjacency matrix from the lower triangular part.

        Args:
            self: the object instance
            lower: the lower triangular matrix

        Returns:
            batch_matrici_diagonali: the recovered full adjacency matrix
        """
        batch_matrici_diagonali = (
            lower + lower.transpose(-2, -1) - torch.diag_embed(lower.diagonal(dim1=-2, dim2=-1))
        )
        return batch_matrici_diagonali

    def permute_adj(self, adj, curr_ind, target_ind):
        """Permute adjacency matrix.
        The target_ind (connectivity) should be permuted to the curr_ind position.
        """
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int32)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes), device=self.device)
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

    def edge_similarity_matrix(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
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
                                * sim_func(matching_features[i], matching_features_recon[a])
                            )
                        except:
                            S[i, i, a, a] = 0

                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = torch.abs(adj[i, j] - adj_recon[a, b])
        return S

    def mpm(self, x_init, S, max_iters=10):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes, device=self.device)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :]) for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim

            norm = torch.norm(x_new)

            x = x_new / norm

        return x

    def forward(self, nodes_features):
        graph_h = nodes_features.reshape(
            -1, self.max_num_nodes * self.num_nodes_features
        )  # spiano la matrice delle features dei nodi

        # VAE:
        z, z_mu, z_lsgms = self.encoder(graph_h)
        h_decode, node_recon_features, edges_recon_features = self.decoder(z)

        # reshape dell'output della VAE in modo da ottenere risultati in forma matriciale
        node_recon_features = node_recon_features.view(-1, self.max_num_nodes, self.num_nodes_features)

        edges_recon_features = edges_recon_features.view(-1, self.max_num_edges, self.num_edges_features)

        # softmax in modo da avere valori probabilistici per la matrice delle features degli edges
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
        recon_adj_lower = self.recover_adj_lower(adj_recon_vector)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # LENTISSIMOO...
        for i in range(adj_recon_vector.shape[0]):

            adj_wout_diagonal = adj_true[i][upper_triangular_indices[0], upper_triangular_indices[1]]
            adj_mask = adj_wout_diagonal.repeat(edges_recon.shape[2], 1).T
            masked_edges_recon_features = edges_recon[i] * adj_mask
            edges_recon_features_total[i] = masked_edges_recon_features.reshape(-1, edges_recon.shape[2])

            S = self.edge_similarity_matrix(
                adj_true[i],
                recon_adj_tensor[i],
                edges_true[i],
                edges_recon[i],
                self.deg_feature_similarity,
            )

            init_assignment = (
                torch.ones(self.max_num_nodes, self.max_num_nodes, device=self.device) * init_corr
            )
            assignment = self.mpm(init_assignment, S)

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.detach().cpu().numpy())
            # Algoritmo ungherese implementato in torch per velocizzare le operazioni e fare tutto su gpu
            # row_ind, col_ind = hungarian_algorithm(assignment)

            adj_permuted = self.permute_adj(adj_true[i], row_ind, col_ind)
            adj_permuted_vectorized[i] = adj_permuted[
                torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
            ]

        adj_recon_loss = F.binary_cross_entropy(adj_recon_vector, adj_permuted_vectorized)

        loss_kl = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes

        loss_edge = F.mse_loss(edges_recon_features_total, edges_true)
        loss_node = F.mse_loss(node_recon, node_true)

        penalization_term = 1
        loss = adj_recon_loss + penalization_term * loss_kl + loss_edge + loss_node

        return loss, adj_recon_loss, loss_kl, loss_edge, loss_node

    def generate(self, z, treshold_adj=0.50, treshold_diag=0.50):

        with torch.no_grad():
            h_decode, output_node_features, output_edge_features = self.decoder(z)

            output_node_features = output_node_features.view(-1, self.max_num_nodes, self.num_nodes_features)

            output_edge_features = output_edge_features.view(-1, self.max_num_edges, self.num_edges_features)

            # degli edge faccio la softmax perchè erano codificati in one-hot
            output_edge_features = F.softmax(output_edge_features, dim=2)

            # della matrice di adiacenza faccio la sigmoid per schiacciare i valori tra 0 e 1
            out = torch.sigmoid(h_decode)

            # ricostruisco la matrice di adiacenza
            recon_adj_lower = self.recover_adj_lower(out)
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

            # Trova gli indici delle righe e delle colonne con valori diagonali superiori a treshold_diag
            diagonal_values = torch.diagonal(recon_adj_tensor, dim1=-2, dim2=-1)
            indices_bool = diagonal_values > treshold_diag

            # Crea la maschera booleana per selezionare le righe e le colonne
            mask = indices_bool.unsqueeze(1) & indices_bool.unsqueeze(2)

            # Seleziona le righe e le colonne
            selected_matrices = recon_adj_tensor * mask

            # Ottenere le dimensioni del tensore
            _, matrix_size, _ = selected_matrices.size()

            # Creare una maschera identità per le diagonali
            diagonal_mask = torch.eye(matrix_size, device=self.device).bool()

            # Moltiplicare ogni matrice per la maschera identità invertita
            result_tensor = selected_matrices * (~diagonal_mask).unsqueeze(0)

            # arrotondo la matrice adiacente ad una treshold definita
            recon_adj_tensor_rounded = torch.round(result_tensor + (0.5 - treshold_adj))

            # conto il numeri di 1 per ogni matrice
            n_one = torch.sum(recon_adj_tensor_rounded, dim=(1, 2)) // 2

            sotto_matrice = F.softmax(output_node_features[:, :, 5:9], dim=2)
            output_node_features[:, :, 5:9] = sotto_matrice.squeeze_()

            return (
                recon_adj_tensor_rounded,
                output_node_features,
                output_edge_features,
                n_one,
            )
