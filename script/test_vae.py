import argparse
import os
from datetime import datetime

import numpy as np


import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from rdkit.Chem import Draw


# ---
from GraphVAE.model_graphvae import GraphVAE
from data_graphvae import load_QM9

from utils import (
    pit,
    load_from_checkpoint,
    save_checkpoint,
    log_metrics,
    latest_checkpoint,
    set_seed,
)

from evaluate import calc_metrics


def build_model(
    max_num_nodes: int = 0,
    max_num_edges: int = 0,
    num_nodes_features: int = 15,
    num_edges_features: int = 4,
    len_num_features: int = 8,
    latent_dimension=5,
    device=torch.device("cpu"),
):

    # out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    # print(num_features)

    input_dim = len_num_features
    model = GraphVAE(
        input_dim,
        latent_dimension,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        device=device,
    ).to(device)
    return model


def test(model, val_loader, latent_dimension, device):
    model.eval()
    smile_pred = []
    smile_true = []
    with torch.no_grad():
        for idx, data in pit(
            enumerate(val_loader),
            total=len(val_loader),
            color="red",
            desc="Validating",
        ):

            z = torch.rand(1, latent_dimension).to(device)
            _, _, _, smile, mol = model.generate(z, smile=True)
            smile_pred.append(smile)
            smile_true.append(data["smiles"])

    calc_metrics(smile_true, smile_pred)


if __name__ == "__main__":

    # loading dataset
    num_examples = 10000
    batch_size = 5
    num_nodes_features = 17
    num_edges_features = 4
    max_num_nodes = 9
    latent_dimension = 5
    checkpoints_dir = "checkpoints"
    learning_rate = 0.001
    LR_milestones = [500, 1000]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD DATASET QM9:
    (
        _,
        _,
        train_dataset_loader,
        test_dataset_loader,
        val_dataset_loader,
        max_num_nodes,
    ) = load_QM9(
        max_num_nodes, num_examples, batch_size, dataset_split_list=(0.1, 0.8, 0.1)
    )

    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    print("-------- Testing --------")
    print("Test set: {}".format(len(test_dataset_loader) * batch_size))
    print("max num edges:", max_num_edges)
    print("max num nodes:", max_num_nodes)
    print("num edges features", num_edges_features)
    print("num nodes features", num_nodes_features)

    model = build_model(
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        len_num_features=max_num_nodes,
        latent_dimension=latent_dimension,
        device=device,
    )

    # Checkpoint load
    if os.path.isdir(checkpoints_dir):
        print(f"trying to load latest checkpoint from directory {checkpoints_dir}")
        checkpoint = latest_checkpoint(checkpoints_dir, "checkpoint")

    if checkpoint is not None and os.path.isfile(checkpoint):
        steps_saved, epochs_saved, train_date_loss = load_from_checkpoint(
            checkpoint,
            model,
        )

    test(model, test_dataset_loader, latent_dimension, device)
