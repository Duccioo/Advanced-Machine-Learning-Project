import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils

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
    # Caricamento del dataset MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=128,
        shuffle=True,
    )

    # Inizializzazione del modello e dell'ottimizzatore
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 10

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

    print("provo a generare una nuova immagine!")

    # Generazione di un vettore di rumore casuale
    z = torch.randn(1, 20)

    # Generazione dell'immagine dal vettore di rumore
    with torch.no_grad():
        sample = model.decode(z).cpu()

    # Visualizzazione dell'immagine generata
    vutils.save_image(sample.view(1, 1, 28, 28), "generated_image.png")
