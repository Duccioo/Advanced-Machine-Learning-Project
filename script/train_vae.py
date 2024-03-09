import argparse
import os
from datetime import datetime
import json

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from rdkit.Chem import Draw
from tqdm import tqdm

# ---
from GraphVAE.model_graphvae import GraphVAE
from data_graphvae import load_QM9
from utils import (
    load_from_checkpoint,
    save_checkpoint,
    log_metrics,
    latest_checkpoint,
    set_seed,
    graph_to_mol,
    generate_unique_id,
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
    # LR_milestones = [int(epochs * 0.3), int(epochs * 0.6)]
    LR_milestones = [500, 1000]
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    epochs_saved = 0
    checkpoint = None
    steps_saved = 0
    running_steps = 0
    val_accuracy = 0
    train_date_loss = []
    validation = []

    latent_dimension = 9

    # Checkpoint load
    if os.path.isdir(checkpoints_dir):
        print(f"trying to load latest checkpoint from directory {checkpoints_dir}")
        checkpoint = latest_checkpoint(checkpoints_dir, "checkpoint")

    if checkpoint is not None and os.path.isfile(checkpoint):
        data_saved = load_from_checkpoint(checkpoint, model, optimizer, scheduler)
        steps_saved = data_saved["step"]
        epochs_saved = data_saved["epoch"]
        train_date_loss = data_saved["loss"]
        validation = data_saved["other"]
        print(f"start from checkpoint at step {steps_saved} and epoch {epochs_saved}")

    p_bar_epoch = tqdm(range(epochs), desc="epochs", position=0)
    for epoch in p_bar_epoch:
        if epoch < epochs_saved:
            continue

        model.train()
        running_loss = 0.0

        # BATCH FOR LOOP
        for i, data in tqdm(
            enumerate(train_loader), total=len(train_loader), position=1, leave=False
        ):
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
            running_steps += 1

            train_date_loss.append((datetime.now(), loss.item()))

        # Validation each epoch:
        model.eval()
        smiles_pred = []
        smiles_true = []
        with torch.no_grad():
            p_bar_val = tqdm(
                val_loader,
                position=1,
                leave=False,
                total=len(val_loader),
                colour="yellow",
                desc="Validation",
            )
            for batch_val in p_bar_val:
                # print("-----")
                z = torch.rand(len(batch_val["smiles"]), latent_dimension).to(device)
                (recon_adj, recon_node, recon_edge, n_one) = model.generate(z, 0.5, 0.4)
                for index_val, elem in enumerate(batch_val["smiles"]):
                    if n_one[index_val] == 0:
                        mol = None
                        smile = None
                    else:
                        mol, smile = graph_to_mol(
                            recon_adj[index_val].cpu(),
                            recon_node[index_val],
                            recon_edge[index_val].cpu(),
                            False,
                            True,
                        )

                    smiles_pred.append(smile)
                    smiles_true.append(elem)

            validity_percentage, _, _ = calc_metrics(smiles_true, smiles_pred)
            validation.append([validity_percentage] * len(train_loader))
            p_bar_val.write(str(validity_percentage))

        p_bar_epoch.write(
            f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )
        p_bar_epoch.write("Sto salvando il modello...")
        save_checkpoint(
            checkpoints_dir,
            f"checkpoint_{epoch+1}",
            model,
            running_steps,
            epoch + 1,
            train_date_loss,
            optimizer,
            scheduler,
            other=validation,
        )

    print("logging")

    train_loss = [x[1] for x in train_date_loss]
    date = [x[0] for x in train_date_loss]

    log_metrics(
        epochs,
        total_batch=len(train_loader),
        train_loss=train_loss,
        val_accuracy=sum(validation, []),
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
        batch_size=1,
        num_workers=1,
        max_num_nodes=4,
        num_examples=15,
        latent_dimension=9,
        epochs=5,
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

    # loading dataset
    num_nodes_features = 17
    num_edges_features = 4
    max_num_nodes = 9

    hyper_params = []

    # LOAD DATASET QM9:
    (_, _, train_dataset_loader, _, val_dataset_loader, max_num_nodes_dataset) = (
        load_QM9(
            max_num_nodes,
            prog_args.num_examples,
            prog_args.batch_size,
            dataset_split_list=(0.7, 0.1, 0.2),
        )
    )
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    print("-------- TRAINING: --------")

    print(
        "Training set: {}".format(
            len(train_dataset_loader) * train_dataset_loader.batch_size
        )
    )
    print(
        "Validation set : {}".format(
            len(val_dataset_loader) * val_dataset_loader.batch_size
        )
    )

    print("max num nodes setted:", max_num_nodes)
    print("max num nodes in dataset:", max_num_nodes_dataset)
    print("max theoretical edges:", max_num_edges)
    print("num edges features", num_edges_features)
    print("num nodes features", num_nodes_features)

    # set up the model:
    model = build_model(
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        len_num_features=max_num_nodes,
        latent_dimension=prog_args.latent_dimension,
        device=device,
    )
    # training:
    train(
        prog_args,
        train_dataset_loader,
        val_dataset_loader,
        model,
        epochs=prog_args.epochs,
        device=device,
    )

    hyper_params.append(
        {
            "num_nodes_features": num_nodes_features,
            "num_edges_features": num_edges_features,
            "max_num_nodes": max_num_nodes,
            "max_num_edges": max_num_edges,
            "latent_dimension": prog_args.latent_dimension,
            "num_examples": prog_args.num_examples,
            "batch_size": prog_args.batch_size,
        }
    )

    model_code = generate_unique_id(list(hyper_params[0].values()), 5)

    # salvo gli iperparametri:
    json_path = f"hyperparams_{model_code}.json"

    # Caricamento dei dati dal file JSON
    with open(json_path, "w") as file:
        json.dump(hyper_params, file)

    # salvo il modello finale:
    model_path = f"final_model_{model_code}.pth"
    torch.save(model.state_dict(), model_path)

    # ---- INFERENCE ----
    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, prog_args.latent_dimension).to(device)

    # Generazione del grafo dal vettore di rumore
    with torch.no_grad():
        adj, features_nodes, features_edges, _ = model.generate(z, 0.50, 0.50)

    rounded_adj_matrix = torch.round(adj)

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


if __name__ == "__main__":
    main()
