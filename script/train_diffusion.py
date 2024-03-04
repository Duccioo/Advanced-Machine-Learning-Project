import os


import torch.nn.functional as F
import torch
from torch.optim import Adam


# ---
from LatentDiffusion.model_latent import SimpleUnet
from LatentDiffusion.model_VAE import MLP_VAE_plain_ENCODER, MLP_VAE_plain_DECODER
from data_graphvae import load_QM9
from utils import set_seed


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
    return F.mse_loss(noise, noise_pred, reduce="sum")


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


if __name__ == "__main__":

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dimension = 6
    num_nodes_features = 17
    num_edges_features = 4
    latent_dim = 5
    max_num_nodes = 6
    max_num_edges = max_num_nodes * (max_num_nodes - 1) // 2
    BATCH_SIZE = 5
    NUM_EXAMPLES = 1000
    epochs = 20  # Try more!
    learning_rate = 0.01
    file_encoder = os.path.join("model_saved", "encoder.pth")
    file_decoder = "decoder.pth"

    encoder = MLP_VAE_plain_ENCODER(
        input_dimension * num_nodes_features,
        latent_dim,
        device=device,
    )
    decoder = MLP_VAE_plain_DECODER(
        input_dim=input_dimension,
        latent_dim=latent_dim,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_nodes_features=num_nodes_features,
        num_edges_features=num_edges_features,
        device=device,
    )
    
    # TODO:
    # 1. LOAD CHECKPOINT FOR ENCODER AND DECODER
    # ...
    encoder.load(file_encoder)
    # decoder.load(file_decoder)

    encoder.to(device)
    decoder.to(device)

    # tanto l'encoder e il decoder non vanno allenati:
    encoder.eval()
    decoder.eval()

    model = SimpleUnet(latent_dim)

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # LOAD DATASET QM9:
    _, _, train_dataset_loader, val_dataset_loader, max_num_nodes = load_QM9(
        max_num_nodes, NUM_EXAMPLES, BATCH_SIZE
    )

    # -- TRAINING -- :
    print("\nStart training...")
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataset_loader):
            optimizer.zero_grad()

            t = torch.randint(
                0, T, (len(batch["features_nodes"]),), device=device
            ).long()
            # print(batch["smiles"][0])
            features_nodes = batch["features_nodes"].float().to(device)
            graph_h = features_nodes.reshape(-1, input_dimension * num_nodes_features)
            # features_edges = batch["features_edges"].float().to(device)
            # adj_input = batch["adj"].float().to(device)
            # print(graph_h.shape)
            # print(features_nodes.shape)
            batch_latent, _, _ = encoder(graph_h)
            loss = get_loss(model, batch_latent, t)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

    # -- INFERENCE -- :
    print("INFERENCE!")
    with torch.no_grad():
        model.eval()

        # sampling di z casuale:
        z = torch.randn(1, latent_dim).to(device)
        z = sample_timestep(z, torch.tensor([0], device=device), model).to(device)

        _, _, _, smile, recon_mol = decoder(z, smile=True)
        print(smile)
