import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils

import argparse

# from torch.autograd import Variable


# Definizione del modello VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Funzione di perdita
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument(
        "--model_name", type=str, default="models/MNIST_VAE.pth", dest="model_name"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="data", dest="download_dataset_folder"
    )
    parser.add_argument("--batch", type=int, default=64, dest="batch_size")
    parser.add_argument(
        "--do_gen", action="store_true", default=False, dest="do_generation"
    )
    parser.add_argument("--do_train", action="store_true", default=False)
    args = parser.parse_args()

    # Caricamento del dataset

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.download_dataset_folder,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Inizializzazione del modello e dell'ottimizzatore
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 20

    if args.do_train is True:
        # Allenamento del modello
        model.train()
        for epoch in range(EPOCHS):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print(
                "Epoch: {} Average loss: {:.4f}".format(
                    epoch, train_loss / len(train_loader.dataset)
                )
            )

        print("Allenamento completato!")

        # Salva il modello su file
        torch.save(model.state_dict(), args.model_name)

    if args.do_generation is True and args.do_train is False:
        # provo a caricarmi il modello già addestrato
        try:
            model.load_state_dict(torch.load(args.model_name))
        except:
            print("model not found!")
            exit()
    elif args.do_generation is False:
        exit()

    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, 20)

    # Generazione dell'immagine dal vettore di rumore
    with torch.no_grad():
        sample = model.decode(z).cpu()

    # Visualizzazione dell'immagine generata
    vutils.save_image(sample.view(1, 1, 28, 28), "generated_image.png")
