import argparse
import os
from datetime import datetime
import random

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# ---
from GraphVAE.model_graphvae import GraphVAE
from utils.data_graphvae import load_QM9
from utils import (
    load_from_checkpoint,
    save_checkpoint,
    log_metrics,
    latest_checkpoint,
    set_seed,
    graph_to_mol,
    Summary,
)
from evaluate import calc_metrics


def validation(model: GraphVAE, val_loader, device, treshold_adj: float = 0.5, treshold_diag: float = 0.5):
    model.eval()
    smiles_pred = []
    smiles_true = []
    with torch.no_grad():
        p_bar_val = tqdm(
            val_loader, position=1, leave=False, total=len(val_loader), colour="yellow", desc="Validation"
        )
        for batch_val in p_bar_val:
            z = torch.rand(len(batch_val["smiles"]), model.embedding_size).to(device)
            (recon_adj, recon_node, recon_edge, n_one) = model.generate(z, treshold_adj, treshold_diag)
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
        return validity_percentage


def train(
    learning_rate,
    train_loader,
    val_loader,
    model,
    epochs=50,
    checkpoints_dir="checkpoints",
    logs_dir="logs",
    device=torch.device("cpu"),
):
    # LR_milestones = [int(epochs * 0.3), int(epochs * 0.6)]
    LR_milestones = [500, 1000]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=learning_rate)

    logs_dir_csv = os.path.join(logs_dir, "metric_training.csv")
    logs_dir_plot = os.path.join(logs_dir, "plot_training.png")

    epochs_saved = 0
    checkpoint = None
    steps_saved = 0
    running_steps = 0
    val_accuracy = 0
    train_date_loss = []
    train_list_losses = []
    validation_saved = []

    # Checkpoint load
    if os.path.isdir(checkpoints_dir):
        print(f"trying to load latest checkpoint from directory {checkpoints_dir}")
        checkpoint = latest_checkpoint(checkpoints_dir, "checkpoint")

    if checkpoint is not None and os.path.isfile(checkpoint):
        data_saved = load_from_checkpoint(checkpoint, model, optimizer, scheduler)
        steps_saved = data_saved["step"]
        epochs_saved = data_saved["epoch"]
        train_date_loss = data_saved["loss"]
        validation_saved = data_saved["other"]
        print(f"start from checkpoint at step {steps_saved} and epoch {epochs_saved}")

    p_bar_epoch = tqdm(range(epochs), desc="epochs", position=0)
    for epoch in p_bar_epoch:
        if epoch < epochs_saved:
            continue

        model.train()
        running_loss = 0.0

        # BATCH FOR LOOP
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), position=1, leave=False):
            features_nodes = data["features_nodes"].float().to(device)
            features_edges = data["features_edges"].float().to(device)
            adj_input = data["adj"].float().to(device)

            adj_vec, mu, var, node_recon, edge_recon = model(features_nodes)

            loss, adj_recon_loss, loss_kl, loss_edge, loss_node = model.loss(
                adj_input, adj_vec, features_nodes, node_recon, features_edges, edge_recon, mu, var
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_steps += 1

            train_date_loss.append((datetime.now(), loss.item()))
            train_list_losses.append(
                [loss.item(), adj_recon_loss.item(), loss_kl.item(), loss_edge.item(), loss_node.item()]
            )

        # Validation each epoch:
        validity_percentage = validation(model, val_loader, device, 0.3, 0.2)
        validation_saved.append([validity_percentage] * len(train_loader))
        print(str(validity_percentage))

        p_bar_epoch.write(
            f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )
        p_bar_epoch.write("Saving checkpoint...")
        save_checkpoint(
            checkpoints_dir,
            f"checkpoint_{epoch+1}",
            model,
            running_steps,
            epoch + 1,
            train_date_loss,
            optimizer,
            scheduler,
            other=validation_saved,
        )

    print("logging")

    train_loss = [x[1] for x in train_date_loss]
    train_date = [x[0] for x in train_date_loss]

    train_adj_recon_loss = [x[1] for x in train_list_losses]
    train_kl_loss = [x[2] for x in train_list_losses]
    train_edge_loss = [x[3] for x in train_list_losses]
    train_node_loss = [x[4] for x in train_list_losses]

    train_dict_losses = {
        "train_total_loss": train_loss,
        "train_adj_recon_loss": train_adj_recon_loss,
        "train_kl_loss": train_kl_loss,
        "train_edge_loss": train_edge_loss,
        "train_node_loss": train_node_loss,
    }

    validation_accuracy = sum(validation_saved, [])

    log_metrics(
        epochs,
        total_batch=len(train_loader),
        train_total_loss=train_loss,
        train_dict_losses=train_dict_losses,
        val_accuracy=validation_accuracy,
        date=train_date,
        title="Training Loss",
        plot_save=True,
        plot_file_path=logs_dir_plot,
        log_file_path=logs_dir_csv,
    )


def count_edges(adj_matrix):
    num_edges = torch.sum(adj_matrix) / 2  # Diviso per 2 perché la matrice adiacente è simmetrica
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

    parser.add_argument("--num_examples", type=int, dest="num_examples", help="Number of examples")
    parser.add_argument("--latent_dimension", type=int, dest="latent_dimension", help="Latent Dimension")
    parser.add_argument("--epochs", type=int, dest="epochs", help="Number of epochs")
    parser.add_argument(
        "--train_percentage", type=int, dest="train_dataset_percentage", help="Train dataset percentage"
    )
    parser.add_argument(
        "--test_percentage", type=int, dest="test_dataset_percentage", help="Train dataset percentage"
    )
    parser.add_argument(
        "--val_percentage", type=int, dest="val_dataset_percentage", help="Train dataset percentage"
    )
    parser.add_argument("--device", type=str, dest="device", help="cuda or cpu")

    dataset_example = 7500

    parser.set_defaults(
        training__lr=0.001,
        dataset__batch_size=15,
        dataset__num_workers=1,
        dataset__max_num_nodes=9,
        dataset__num_examples=6000,
        model__latent_dimension=9,
        training__epochs=5,
        dataset__train_percentage=0.7,
        dataset__test_percentage=0.0,
        dataset__val_percentage=0.3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def main():

    set_seed(42)

    prog_args = arg_parse()

    device = prog_args.device

    experiment_model_type = "GRAPH VAE"
    model_folder = "models"
    experiment_folder = os.path.join(model_folder, "logs_GraphVAE_v2_" + str(prog_args.dataset__num_examples))

    # loading dataset
    max_num_nodes = prog_args.dataset__max_num_nodes

    # LOAD DATASET QM9:
    (_, dataset_pad, train_dataset_loader, _, val_dataset_loader, max_num_nodes_dataset) = load_QM9(
        max_num_nodes,
        prog_args.dataset__num_examples,
        prog_args.dataset__batch_size,
        dataset_split_list=(
            prog_args.dataset__train_percentage,
            prog_args.dataset__test_percentage,
            prog_args.dataset__val_percentage,
        ),
    )
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    num_nodes_features = dataset_pad[0]["features_nodes"].shape[1]
    num_edges_features = dataset_pad[0]["features_edges"].shape[1]

    training_effective_size = len(train_dataset_loader) * train_dataset_loader.batch_size
    validation_effective_size = len(val_dataset_loader) * val_dataset_loader.batch_size

    print("-------- EXPERIMENT INFO --------")

    print("Training set: {}".format(training_effective_size))
    print("Validation set : {}".format(validation_effective_size))

    print("max num nodes in dataset:", max_num_nodes_dataset)
    print("max num nodes setted:", max_num_nodes)
    print("max theoretical edges:", max_num_edges)
    print("num edges features", num_edges_features)
    print("num nodes features", num_nodes_features)

    # prima di iniziare il training creo la cartella in cui salvere i checkpoints, modello, iperparametri, etc...
    model__hyper_params = []
    dataset__hyper_params = []

    model__hyper_params.append(
        {
            "num_nodes_features": num_nodes_features,
            "num_edges_features": num_edges_features,
            "max_num_nodes": max_num_nodes,
            "max_num_edges": max_num_edges,
            "latent_dimension": prog_args.model__latent_dimension,
        }
    )

    dataset__hyper_params.append(
        {
            "learning_rate": prog_args.training__lr,
            "epochs": prog_args.training__epochs,
            "num_examples": prog_args.dataset__num_examples,
            "batch_size": prog_args.dataset__batch_size,
            "training_percentage": prog_args.dataset__train_percentage,
            "test_percentage": prog_args.dataset__test_percentage,
            "val_percentage": prog_args.dataset__val_percentage,
            "number_val_examples": validation_effective_size,
            "number_train_examples": training_effective_size,
        }
    )

    summary = Summary(experiment_folder, experiment_model_type)
    summary.save_model_json(model__hyper_params)
    summary.save_summary_training(dataset__hyper_params, model__hyper_params, random.choice(dataset_pad))

    print("-------- TRAINING --------")

    # set up the model:
    model = GraphVAE(
        latent_dim=prog_args.model__latent_dimension,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        device=device,
    )

    # training:
    train(
        prog_args.training__lr,
        train_dataset_loader,
        val_dataset_loader,
        model,
        epochs=prog_args.training__epochs,
        device=device,
        checkpoints_dir=summary.directory_checkpoint,
        logs_dir=summary.directory_log,
    )

    # salvo il modello finale:
    final_model_name = "trained_GraphVAE_FINAL.pth"
    final_model_path = os.path.join(summary.directory_base, final_model_name)
    torch.save(model.state_dict(), final_model_path)

    # ---- INFERENCE ----
    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, prog_args.model__latent_dimension).to(device)

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
