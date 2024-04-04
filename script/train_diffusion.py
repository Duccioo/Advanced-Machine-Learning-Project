import os
import json
import random
from datetime import datetime

from tqdm import tqdm

import torch.nn.functional as F
import torch
from torch.optim import Adam


# ---
from LatentDiffusion.model_latent import SimpleUnet
from GraphVAE.model_base import MLP_VAE_plain_ENCODER
from GraphVAE.model_graphvae import GraphVAE
from utils.data_graphvae import load_QM9
from evaluate import calc_metrics
from utils import (
    set_seed,
    graph_to_mol,
    load_from_checkpoint,
    save_checkpoint,
    log_metrics,
    latest_checkpoint,
)
from utils.summary import Summary


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    # print("Forward Diffusion Sampling:")
    # print(x_0.shape)
    # print(sqrt_alphas_cumprod_t.shape)

    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise_pred, noise)


@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def load_GraphVAE(model_folder: str = "", device="cpu"):
    """
    A function to load a GraphVAE model from the specified model folder.

    Parameters:
    - model_folder (str): The path to the folder containing the model files.
    - device (str): The device where the model will be loaded (default is "cpu").

    Returns:
    - model_GraphVAE (GraphVAE): The loaded GraphVAE model.
    - encoder (MLP_VAE_plain_ENCODER): The encoder model associated with the GraphVAE.
    - hyper_params (dict): The hyperparameters used for the model.
    """

    json_files = [file for file in os.listdir(model_folder) if file.endswith(".json")]
    # Cerca i file con estensione .pth
    pth_files = [file for file in os.listdir(model_folder) if file.endswith(".pth")]

    if json_files:
        hyper_file_json = os.path.join(model_folder, json_files[0])
    else:
        hyper_file_json = None

    if pth_files:
        model_file_pth = os.path.join(model_folder, pth_files[0])
    else:
        model_file_pth = None

    with open(hyper_file_json, "r") as file:
        dati = json.load(file)
        hyper_params = dati[0]

    model_GraphVAE = GraphVAE(
        hyper_params["latent_dimension"],
        max_num_nodes=hyper_params["max_num_nodes"],
        max_num_edges=hyper_params["max_num_edges"],
        num_nodes_features=hyper_params["num_nodes_features"],
        num_edges_features=hyper_params["num_edges_features"],
        device=device,
    )

    data_model = torch.load(model_file_pth, map_location="cpu")
    model_GraphVAE.load_state_dict(data_model)

    model_GraphVAE.to(device)

    encoder = MLP_VAE_plain_ENCODER(
        hyper_params["max_num_nodes"] * hyper_params["num_nodes_features"],
        hyper_params["latent_dimension"],
        device=device,
    ).to(device)

    encoder.load_state_dict(model_GraphVAE.encoder.state_dict())

    return model_GraphVAE, encoder, hyper_params


def validation(
    model_diffusion: SimpleUnet,
    model_vae: GraphVAE,
    hyperparams,
    val_loader,
    device,
    treshold_adj: float = 0.5,
    treshold_diag: float = 0.5,
):
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
            z = torch.rand(
                len(batch_val["smiles"]), hyperparams["latent_dimension"]
            ).to(device)
            z = sample_timestep(
                z, torch.tensor([0], device=device), model_diffusion
            ).to(device)
            (recon_adj, recon_node, recon_edge, n_one) = model_vae.generate(
                z, treshold_adj, treshold_diag
            )
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
    hyperparams,
    train_loader,
    val_loader,
    encoder,
    model_diffusion,
    model_vae,
    epochs=50,
    checkpoints_dir="checkpoints",
    logs_dir="logs",
    device=torch.device("cpu"),
):

    optimizer = Adam(model.parameters(), lr=learning_rate)

    logs_dir_csv = os.path.join(logs_dir, "metric_training.csv")
    logs_dir_plot = os.path.join(logs_dir, "plot_training.png")

    epochs_saved = 0
    checkpoint = None
    steps_saved = 0
    running_steps = 0
    val_accuracy = 0
    train_date_loss = []
    validation_saved = []

    # Checkpoint load
    if os.path.isdir(checkpoints_dir):
        print(f"trying to load latest checkpoint from directory {checkpoints_dir}")
        checkpoint = latest_checkpoint(checkpoints_dir, "checkpoint")

    if checkpoint is not None and os.path.isfile(checkpoint):
        data_saved = load_from_checkpoint(checkpoint, model, optimizer)
        steps_saved = data_saved["step"]
        epochs_saved = data_saved["epoch"]
        train_date_loss = data_saved["loss"]
        validation_saved = data_saved["other"]
        print(f"start from checkpoint at step {steps_saved} and epoch {epochs_saved}")

    p_bar_epoch = tqdm(range(epochs), desc="epochs", position=0)
    for epoch in p_bar_epoch:
        if epoch < epochs_saved:
            continue

        model_diffusion.train()
        running_loss = 0.0

        # BATCH FOR LOOP
        for i, data in tqdm(
            enumerate(train_loader), total=len(train_loader), position=1, leave=False
        ):
            optimizer.zero_grad()

            t = torch.randint(
                0, T, (len(data["features_nodes"]),), device=device
            ).long()
            # print(data["smiles"][0])
            features_nodes = data["features_nodes"].float().to(device)
            graph_h = features_nodes.reshape(
                -1, hyperparams["max_num_nodes"] * hyperparams["num_nodes_features"]
            )
            # features_edges = batch["features_edges"].float().to(device)
            # adj_input = batch["adj"].float().to(device)
            # print(graph_h.shape)
            # print(features_nodes.shape)
            batch_latent, _, _ = encoder(graph_h)
            loss = get_loss(model_diffusion, batch_latent, t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_steps += 1

            train_date_loss.append((datetime.now(), loss.item()))

        # Validation each epoch:
        validity_percentage = validation(
            model_diffusion, model_vae, hyperparams, val_loader, device, 0.3, 0.2
        )
        validation_saved.append([validity_percentage] * len(train_loader))

        p_bar_epoch.write(
            f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {validity_percentage:.2f}%"
        )
        p_bar_epoch.write("Saving checkpoint...")
        if epoch % 5 == 0 and epoch != 0:
            save_checkpoint(
                checkpoints_dir,
                f"checkpoint_{epoch+1}",
                model,
                running_steps,
                epoch + 1,
                train_date_loss,
                optimizer,
                other=validation_saved,
            )

    print("logging")

    train_loss = [x[1] for x in train_date_loss]
    train_date = [x[0] for x in train_date_loss]
    validation_accuracy = sum(validation_saved, [])

    log_metrics(
        epochs,
        total_batch=len(train_loader),
        train_loss=train_loss,
        val_accuracy=validation_accuracy,
        date=train_date,
        title="Training Loss",
        plot_save=True,
        plot_file_path=logs_dir_plot,
        log_file_path=logs_dir_csv,
    )


if __name__ == "__main__":

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 64
    NUM_EXAMPLES = 100000
    epochs = 50 
    learning_rate = 0.01
    train_percentage = 0.7
    test_percentage = 0.0
    val_percentage = 0.3

    down_channel = (7, 5, 3)
    time_emb_dim = 6

    experiment_model_type = "Diffusion"
    model_folder = "models"
    experiment_folder = os.path.join(
        model_folder, "logs_Diffusion_" + str(NUM_EXAMPLES)
    )

    folder_GraphVAE = os.path.join(model_folder, "logs_GraphVAE_130")

    decoder, encoder, hyperparams = load_GraphVAE(
        model_folder=folder_GraphVAE, device=device
    )
    hyperparams["down_channel"] = down_channel
    hyperparams["time_emb_dim"] = time_emb_dim

    # tanto l'encoder e il decoder non vanno allenati:
    encoder.eval()
    decoder.eval()

    model = SimpleUnet(hyperparams["latent_dimension"], down_channel, time_emb_dim)

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # LOAD DATASET QM9:
    (
        _,
        dataset_pad,
        train_dataset_loader,
        test_dataset_loader,
        val_dataset_loader,
        max_num_nodes,
    ) = load_QM9(
        hyperparams["max_num_nodes"],
        NUM_EXAMPLES,
        BATCH_SIZE,
        dataset_split_list=(train_percentage, test_percentage, val_percentage),
    )

    training_effective_size = (
        len(train_dataset_loader) * train_dataset_loader.batch_size
    )
    validation_effective_size = len(val_dataset_loader) * val_dataset_loader.batch_size

    dataset__hyper_params = []
    dataset__hyper_params.append(
        {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "num_examples": NUM_EXAMPLES,
            "batch_size": BATCH_SIZE,
            "training_percentage": train_percentage,
            "test_percentage": test_percentage,
            "val_percentage": val_percentage,
            "number_val_examples": validation_effective_size,
            "number_train_examples": training_effective_size,
        }
    )

    summary = Summary(experiment_folder, experiment_model_type)
    summary.save_model_json(hyperparams)
    summary.save_summary_training(
        dataset__hyper_params, hyperparams, random.choice(dataset_pad)
    )

    # -- TRAINING -- :
    print("\nStart training...")
    train(
        learning_rate,
        hyperparams,
        train_dataset_loader,
        val_dataset_loader,
        encoder,
        model,
        decoder,
        epochs,
        checkpoints_dir=summary.directory_checkpoint,
        logs_dir=summary.directory_log,
        device=device,
    )

    # salvo il modello finale:
    final_model_name = "trained_Diffusion_FINAL.pth"
    final_model_path = os.path.join(summary.directory_base, final_model_name)
    torch.save(model.state_dict(), final_model_path)

    # -- INFERENCE -- :
    print("INFERENCE!")
    with torch.no_grad():
        model.eval()

        # sampling di z casuale:
        z = torch.randn(1, hyperparams["latent_dimension"]).to(device)
        z = sample_timestep(z, torch.tensor([0], device=device), model).to(device)

        (recon_adj, recon_node, recon_edge, n_one) = decoder.generate(z, 0.4, 0.3)
        mol, smile = graph_to_mol(
            recon_adj[0].cpu(),
            recon_node[0],
            recon_edge[0].cpu(),
            True,
            True,
        )
        print(smile)
