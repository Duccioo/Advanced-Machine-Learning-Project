import os
import torch
from tqdm import tqdm

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


def test(model, val_loader, latent_dimension, device):
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
            (recon_adj, recon_node, recon_edge, n_one) = model.generate(z, 0.4, 0.30)
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

    print("CALCOLO METRICHE:")
    validity_percentage, uniqueness_percentage, novelty_percentage = calc_metrics(
        smiles_true, smiles_pred
    )

    print(smiles_true[0:15])
    print(smiles_pred[0:15])

    print(f"Validità: {validity_percentage:.2%}")
    print(f"Unicità: {uniqueness_percentage:.2%}")
    print(f"Novità: {novelty_percentage:.2%}")

    print("Numero edge medi predetti: ", edges_medi_pred / len(smiles_pred))
    print("Numero edge medi true: ", edges_medi_true / len(smiles_true))


if __name__ == "__main__":
    set_seed(42)

    # loading dataset
    num_examples = 10000
    batch_size = 10
    num_nodes_features = 17
    num_edges_features = 4
    max_num_nodes = 9
    latent_dimension = 9
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
        data_saved = load_from_checkpoint(
            checkpoint,
            model,
        )

    test(model, test_dataset_loader, latent_dimension, device)
