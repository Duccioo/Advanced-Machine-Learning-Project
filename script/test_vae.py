import os
import torch
from tqdm import tqdm
import argparse
import csv
from datetime import datetime

# ---
from GraphVAE.model_graphvae import GraphVAE
from data_graphvae import load_QM9
from utils import load_from_checkpoint, latest_checkpoint, graph_to_mol, set_seed
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


def test(model, val_loader, latent_dimension, device, treshold_adj, treshold_diag):
    model.eval()
    smiles_pred = []
    smiles_true = []
    edges_medi_pred = 0
    edges_medi_true = 0
    val_pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        colour="red",
        desc="Batch",
        position=0,
        leave=True,
    )
    with torch.no_grad():
        for idx, data in val_pbar:
            # print("-----")
            z = torch.rand(len(data), latent_dimension).to(device)
            (recon_adj, recon_node, recon_edge, n_one) = model.generate(
                z, treshold_adj, treshold_diag
            )
            for idx_data, elem in enumerate(data):
                if n_one[idx_data] == 0:
                    mol = None
                    smile = None
                else:
                    mol, smile = graph_to_mol(
                        recon_adj[idx_data].cpu(),
                        recon_node[idx_data],
                        recon_edge[idx_data].cpu(),
                        False,
                        True,
                    )
                # if smile == "" or smile == None:
                #     val_pbar.write("ERROR impossibile creare Molecola")

                smiles_pred.append(smile)
                smiles_true.append(data["smiles"][idx_data])

                edges_medi_pred += n_one[idx_data]
                edges_medi_true += data["num_edges"][idx_data]

    validity_percentage, uniqueness_percentage, novelty_percentage = calc_metrics(
        smiles_true, smiles_pred
    )

    edges_medi_pred = (edges_medi_pred / len(smiles_pred)).item()
    edges_medi_true = (edges_medi_true / len(smiles_true))[0].item()

    return (
        validity_percentage,
        uniqueness_percentage,
        novelty_percentage,
        edges_medi_pred,
        edges_medi_true,
    )


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")

    parser.add_argument(
        "--treshold_adj",
        dest="treshold_adj",
        type=float,
        help="Treshold dei valori fuori dalla diagonale della matrice adiacente",
    )
    parser.add_argument(
        "--treshold_diag",
        dest="treshold_diag",
        type=float,
        help="Treshold dei valori nella diagonale della matrice adiacente",
    )

    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")

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
        treshold_adj=0.5,
        treshold_diag=0.5,
        lr=0.001,
        batch_size=30,
        num_workers=1,
        max_num_nodes=9,
        num_examples=20000,
        latent_dimension=9,
        epochs=5,
        # device=torch.device("cpu"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def write_csv(file_csv, header, risultati):
    # Controlla se il file esiste
    file_esiste = os.path.isfile(file_csv)

    if len(risultati) < len(header):
        risultati.append(datetime.now().strftime("%Y-%m-%d"))

    # Apertura del file in modalità append
    with open(file_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_esiste:
            # Scrivi l'header se il file non esiste
            writer.writerow(header)
        # Aggiungi i risultati
        writer.writerow(risultati)


if __name__ == "__main__":
    print("~" * 20, "TESTING", "~" * 20)
    set_seed(42)

    args_parsed = arg_parse()

    # loading dataset
    num_nodes_features = 17
    num_edges_features = 4
    num_examples = args_parsed.num_examples
    batch_size = args_parsed.batch_size

    max_num_nodes = args_parsed.max_num_nodes
    latent_dimension = args_parsed.latent_dimension

    checkpoints_dir = "checkpoints"
    learning_rate = args_parsed.lr

    LR_milestones = [500, 1000]

    device = args_parsed.device

    # LOAD DATASET QM9:
    (
        _,
        _,
        train_dataset_loader,
        test_dataset_loader,
        val_dataset_loader,
        max_num_nodes,
    ) = load_QM9(
        max_num_nodes, num_examples, batch_size, dataset_split_list=(0.5, 0.5, 0.0)
    )

    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2

    print("----------------" * 2)
    print("Test set: {}".format(len(test_dataset_loader) * batch_size))
    print("max num edges:", max_num_edges)
    print("max num nodes:", max_num_nodes)
    print("num edges features", num_edges_features)
    print("num nodes features", num_nodes_features)
    print("Treshold Adj:", args_parsed.treshold_adj)
    print("Treshold Diag:", args_parsed.treshold_diag)

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
        data_saved = load_from_checkpoint(
            checkpoint,
            model,
        )

    validity, uniqueness, novelty, edges_pred, edges_true = test(
        model,
        test_dataset_loader,
        latent_dimension,
        device,
        treshold_adj=args_parsed.treshold_adj,
        treshold_diag=args_parsed.treshold_diag,
    )

    print(f"Validità: {validity:.2%}")
    print(f"Unicità: {uniqueness:.2%}")
    print(f"Novità: {novelty:.2%}")

    print("Numero edge medi predetti: ", edges_pred)
    print("Numero edge medi true: ", edges_true)
    header = [
        "Validity %",
        "Uniqueness %",
        "Novelty %",
        "Edges Medi pred",
        "Edges Medi true",
        "Data",
    ]
    results = [validity * 100, uniqueness * 100, novelty * 100, edges_pred, edges_true]
    nome_file = "result_vae.csv"

    write_csv(nome_file, header, results)
