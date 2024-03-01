import argparse
import os
from datetime import datetime

import numpy as np


import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from rdkit.Chem import Draw
from tqdm import tqdm

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


def train(
    args,
    train_loader,
    val_loader,
    model,
    epochs=50,
    checkpoints_dir="checkpoints",
    device=torch.device("cpu"),
):
    LR_milestones = [500, 1000]

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    batch_idx = 0
    epochs_saved = 0
    checkpoint = None
    steps_saved = 0
    running_steps = 0
    val_accuracy = 0
    train_date_loss = []

    # Checkpoint load
    if os.path.isdir(checkpoints_dir):
        print(f"trying to load latest checkpoint from directory {checkpoints_dir}")
        checkpoint = latest_checkpoint(checkpoints_dir, "checkpoint")

    if checkpoint is not None and os.path.isfile(checkpoint):
        steps_saved, epochs_saved, train_date_loss = load_from_checkpoint(
            checkpoint, model, optimizer, scheduler
        )
        print(f"start from checkpoint at step {steps_saved} and epoch {epochs_saved}")

    for epoch in range(0, epochs):
        if epoch < epochs_saved:
            continue

        model.train()
        running_loss = 0.0

        # BATCH FOR LOOP
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            features_nodes = data["features_nodes"].float().to(device)
            features_edges = data["features_edges"].float().to(device)
            adj_input = data["adj"].float().to(device)

            model.zero_grad()

            adj_vec, mu, var, node_recon, edge_recon = model(features_nodes)

            loss = model.loss(
                adj_input,
                adj_vec,
                features_nodes,
                node_recon,
                features_edges,
                edge_recon,
                mu,
                var,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            # Calculate validation accuracy

            running_steps += 1
            train_date_loss.append((datetime.now(), loss.item()))

        print(
            f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )
        print("Sto salvando il modello...")
        save_checkpoint(
            checkpoints_dir,
            f"checkpoint_{epoch}",
            model,
            running_steps,
            epoch + 1,
            train_date_loss,
            optimizer,
            scheduler,
        )

    print("logging")

    train_loss = [x[1] for x in train_date_loss]

    date = [x[0] for x in train_date_loss]
    log_metrics(
        epochs,
        total_batch=len(train_loader),
        train_loss=train_loss,
        date=date,
        title="Training Loss",
        plot_save=True,
    )


def count_edges(adj_matrix):
    num_edges = (
        torch.sum(adj_matrix) / 2
    )  # Diviso per 2 perché la matrice adiacente è simmetrica
    return num_edges


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by training data.",
    )

    parser.add_argument(
        "--num_examples", type=int, dest="num_examples", help="Number of examples"
    )
    parser.add_argument(
        "--latent_dimension", type=int, dest="latent_dimension", help="Latent Dimension"
    )
    parser.add_argument("--epochs", type=int, dest="epochs", help="Number of epochs")
    parser.add_argument("--device", type=str, dest="device", help="cuda or cpu")

    parser.set_defaults(
        lr=0.001,
        batch_size=15,
        num_workers=1,
        max_num_nodes=4,
        num_examples=15000,
        latent_dimension=9,
        epochs=3,
        # device=torch.device("cpu"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def save_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


def main():

    set_seed(42)

    prog_args = arg_parse()

    device = prog_args.device

    filter_apriori_max_num_nodes = -1

    # loading dataset
    num_nodes_features = 17
    num_edges_features = 4
    max_num_nodes = 9

    # LOAD DATASET QM9:
    (
        _,
        _,
        train_dataset_loader,
        test_dataset_loader,
        val_dataset_loader,
        max_num_nodes,
    ) = load_QM9(max_num_nodes, prog_args.num_examples, prog_args.batch_size)
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    print("-------- TRAINING: --------")

    print(
        "training set: {}".format(
            len(train_dataset_loader) * train_dataset_loader.batch_size
        )
    )

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
        latent_dimension=prog_args.latent_dimension,
        device=device,
    )

    train(
        prog_args,
        train_dataset_loader,
        val_dataset_loader,
        model,
        epochs=prog_args.epochs,
        device=device,
    )

    # ---- INFERENCE ----
    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, prog_args.latent_dimension).to(device)
    z = torch.randn(1, prog_args.latent_dimension).to(device)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj, features_nodes, features_edges, smile, mol, _ = model.generate(
            z, smile=True
        )

    rounded_adj_matrix = torch.round(adj)
    # features_nodes = torch.round(features_nodes)
    # features_edges = torch.round(features_edges)
    print("ORIGINAL MATRIX")
    first_molecula = next(iter(train_dataset_loader))
    print(first_molecula["adj"][0])
    print(first_molecula["features_nodes"][0])
    print(first_molecula["smiles"][0])
    print("##" * 10)
    print("Predicted MATRIX")
    print(rounded_adj_matrix)
    print(features_nodes)
    print(features_edges)
    # smiles = data_to_smiles(features_nodes, features_edges, rounded_matrix)
    # print(smiles)

    print("MATCH DELLE MATRICI")
    print(features_edges.shape)
    print(count_edges(rounded_adj_matrix))
    # model.save_vae_encoder("main")

    save_filepath = os.path.join("", "mol_{}.png".format(1))
    print(smile)
    save_png(mol, save_filepath, size=(600, 600))


if __name__ == "__main__":
    main()
