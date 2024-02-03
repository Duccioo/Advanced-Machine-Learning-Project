# Advanced-Machine-Learning-Project

> Progetto che consisterebbe nel realizzare una versione dei diffusion model che usa lo spazio latente.
> In pratica si trattrebbe di generare una rappresentare latente di un grafo, ad esempio usando graphVAE. Poi potremo provare a generare tale rappresentazione
> con un metodo a diffusione. Essendo lo stato latente un semplice vettore, la sua generazione è più semplice, ma il fatto che si debba comporre due metodi da vita ad una cosa >un pò più complicata.

## Models

- Diffusion Model:

  [riguardate le slides sui latent diffusion model per le immagini.](/content/DiffusionModels_prof.pdf)
  L'idea è quella di riapplicare la stessa idea ai grafi.
  A questa vi aggiungo una review sui diffusion model per grafi. Dategli un'occhiata rapida, tenendo conto che i metodi
  menzionati li sono diffusion, ma non nello spazio latente.

  - [A Survey on Graph Diffusion Models: Generative AI in Science for Molecule, Protein and Material](https://arxiv.org/abs/2304.01565)

- GraphVAE model:

  Link all'articolo di GraphVAE che è il variational autoencoder che dovrete usare per creare lo spazio latente:

  - [GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders](https://arxiv.org/abs/1802.03480)

  Per il software, la repo GitHub che vi consiglio di usare è questa:

  - [GraphVAE model baseline Github Repo](https://github.com/JiaxuanYou/graph-generation/tree/master/baselines/graphvae)

  Ci sono anche altri modelli nella stessa repo che potrebbero essere interessanti, ma secondo me conviene partire con questa implementazione di GraphVAE che vi ho linkato.

## Dataset

Per quanto riguarda il dataset (_QM9_), potete comodamente scaricarlo tramite il pacchetto pytorch_geometric, come spiegato a questo link:

- [QM9 DataSet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html)

  E' sufficiente includere questa riga tra le importazioni:
  `from pytorch_geometric.datasets import QM9`

Il dataset _QM9_ preso da _PyTorch Geometric_ è strutturato come un oggetto InMemoryDataset di PyTorch Geometric. L’oggetto contiene i seguenti attributi:

- `data.x`: Matrice di features dei nodi (dimensione `num_nodes x num_node_features`)

- `data.edge_index`: Matrice di adiacenza sparsa che rappresenta gli archi tra i nodi (dimensione `2 x num_edges`)

- `data.edge_attr`: Matrice di features degli archi (dimensione `num_edges x num_edge_features`)

- `data.y`: Tensor contenente la proprietà target per ogni molecola nel dataset (dimensione `num_molecules x num_targets`)

### Visualizzare il dataset:

https://github.com/chainer/chainer-chemistry/blob/master/examples/qm9/qm9_dataset_exploration.ipynb

## Resurces

- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)

  - [Notebook on ImageNet](https://github.com/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb)

- [Hugging Face](https://huggingface.co/fusing/latent-diffusion-text2im-large)

- [Diffusers](https://github.com/huggingface/diffusers)

- [List of papers and repo for Graph Diffusion](https://github.com/ChengyiLIU-cs/Generative-Diffusion-Models-on-Graphs)


## Domande
- Quali metriche usare per confrontare se il grafo generato è buono o no
- Le features dei nodi e degli edge vanno inserite dentro il modello? (soprattutto quelle degli edge)
- python QM9_smiles problema in 98! -> [H]C([H])([H])[N@@H+]1C([H])([H])[C@]1([H])C([H])([H])[H] vs [H]C([H])([H])N1C([H])([H])C1([H])C([H])([H])[H]