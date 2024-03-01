import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
import os


# ---
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import graph_to_mol


class MLP_VAE_plain_ENCODER(nn.Module):

    def __init__(self, h_size, embedding_size, device):
        super(MLP_VAE_plain_ENCODER, self).__init__()
        self.device = device
        self.encode_11 = nn.Linear(h_size, embedding_size).to(device=device)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size).to(device=device)  # lsgms

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(self.device)
        z = eps * z_sgm + z_mu

        return z, z_mu, z_lsgms

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size, e_size, device):
        super(MLP_VAE_plain, self).__init__()

        self.device = device
        self.encoder = MLP_VAE_plain_ENCODER(h_size, embedding_size, device)

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)

        self.relu = nn.ReLU()

        self.decode_1_features = nn.Linear(embedding_size, embedding_size)
        self.decode_2_features = nn.Linear(embedding_size, h_size)

        self.decode_edges_1_features = nn.Linear(embedding_size, embedding_size)
        self.decode_edges_2_features = nn.Linear(embedding_size, e_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, h):

        z, z_mu, z_lsgms = self.encoder(h)

        y, n_features, e_features = self.decoder(z)

        return y, z_mu, z_lsgms, n_features, e_features

    def save_encoder(self, path_to_save_model):
        torch.save(
            self.encoder.state_dict(), os.path.join(path_to_save_model, "encoder.pth")
        )

    def decoder(self, z):
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)

        # decoder for node features
        n_features = self.decode_1_features(z)
        n_features = self.relu(n_features)
        n_features = self.decode_2_features(n_features)

        # decoder for edges features
        e_features = self.decode_edges_1_features(z)
        e_features = self.relu(e_features)
        e_features = self.decode_edges_2_features(e_features)

        return y, n_features, e_features


class MLP_VAE_plain_DECODER(nn.Module):
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
        super(MLP_VAE_plain_DECODER, self).__init__()
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

        # self.feature_mlp = MLP_plain(latent_dim, latent_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def recover_adj_lower(self, l, device=torch.device("cuda")):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes, device=device)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def forward(self, z, device="cpu", smile=False):
        mol = None
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
            print(f"Sotto Matrice::::: {sotto_matrice}")
            print(f"OUTPUT node Features {output_node_features} ")

            indici = torch.argmax(sotto_matrice, dim=2)[0]

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
