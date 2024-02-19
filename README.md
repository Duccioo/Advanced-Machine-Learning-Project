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

  _Altre implementazioni:_

  - [Official pytorch_geometric VGAE implementation](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py)
  - [Another VGAE implementation](https://github.com/DaehanKim/vgae_pytorch/blob/master/model.py)

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

## Question & Answer

1. C'è differenza tra modelli GraphVAE e VGAE?
   <details>

   > GraphVAE e VGAE sono due modelli leggermente diversi, ma appartengono alla stessa classe (graph-based variational autoencoders) e sono del tutto intercambiabili, quindi usate pure quello che vi sembra meglio.
   > Comunque, cambiando dataset, le matrici di adiacenza cambiano. Nella repository che vi ho passato, il file "train.py" carica le matrici di adiacenza dal dataset "enzymes", mentre a voi servono quelle di QM9. Il DataLoader di QM9 che trovate in pytorch_geometric carica i grafi di QM9, ognuno dei quali contiene la sua matrice di adiacenza. Tendenzialmente, il DataLoader di QM9 dovrebbe funzionare con GraphVAE e con VGAE. Se non fosse così, ci sarà da adattare la classe GraphAdjSampler che trovate in "data.py" al formato dati di QM9. GraphAdjSampler infatti è un DataLoader scritto dagli autori della repo (in poche righe di codice) appositamente per il dataset "Enzymes".

  </details>

2. Quali metriche conviene utilizzare per confrontare se il grafo generato è buono o no?
   <details>

   > Per capire se i grafi generati sono buoni o no si può procedere in due modi: valutare Validity, Uniqueness e Novelty, e controllare che le distribuzioni di probabilità siano simili a quelle del training set. Per adesso mi limiterei a Validity, Uniqueness, Novelty. La validity si può valutare col pacchetto python RdKit con una routine che permette di scartare i grafi che violano qualche regola della Chimica e di misurare la percentuale di grafi generati che sono validi. La Uniqueness viene calcolata comparando i grafi generati uno a uno e scartando i doppioni, misurando la percentuale di molecole che non sono copie di altre. Infine, la novelty viene misurata comparando tutti i grafi validi e unici generati con i grafi del training set, ottenendo la percentuale di materiale effettivamente "nuovo" generato dalla rete.

     </details>

3) Le features dei edge vanno inserite dentro il modello?
    <details>

   > Le edge features sono una delle matrici da passare in input al modello: il DataLoader di QM9, una volta integrato nel codice, dovrebbe automaticamente passarle al modello quando si chiama la funzione "train". Se i tipi di archi definiti da Enzymes e da QM9 non sono uguali (come sospetto), ci sta che vada aggiustato un parametro nel codice del modello per far funzionare GraphVAE sulla matrice di QM9.

    </details>

4) Ci siamo accorti, infine, che analizzando il dataset QM9 c'è qualcosa che non torna.
   Un esempio è la molecola numero 23 denominata "gdb_24" con una ground truth smiles = [H]CC#C[NH3+].
   Se noi ci calcoliamo in base alle matrici e alle features presenti nella molecola 23 otteniamo uno smiles = [H]CC#CN.

    <details>

   > Il "problema" delle SMILES è che non sono rappresentazioni univoche: la stessa molecola può essere rappresentata da un sacco di diverse SMILES, a seconda dell'atomo da cui si inizia a scrivere la stringa SMILES e a seconda delle ramificazioni che scegliamo di espandere per prime durante la visita del grafo. Esiste un modo per rendere le SMILES "canoniche", seguendo delle regole che stabiliscono quali ramificazioni espandere per prime ecc..., ma anche in questo caso non si rende univoca la rappresentazione (spesso, esistono più SMILES canoniche per una molecola).
   > Gli ioni possono essere riportati alla molecola con carica neutra corrispondente, per cui [H]CC#C[NH3+] = [H]CC#C[NH2].
   > Inoltre, gli atomi di idrogeno sono spesso superflui nell descrizione di una molecola organica, per cui [H]CC#C[NH3+] = [H]CC#C[NH2] = [H]CC#CN = CC#CN
   > Spesso, anche i generatori di grafi ignorano gli atomi di idrogeno, soprattutto modelli come i GraphVAE che non sono invarianti alle permutazioni dell'ordinamento del grafo. Non considerando gli idrogeni, infatti, si riduce di più della metà il numero di atomi presenti in una molecola, riducendo drasticamente il numero di ordinamenti possibili per il grafo molecolare che la rappresenta. Anche l'addestramento del modello, in termini di memoria occupata e tempo di esecuzione, beneficia molto dell'assenza degli atomi di idrogeno.
   > Se trovate degli idrogeni nei dati di QM9, vi conviene riscrivere il DataLoader in modo da eliminarli (sia dalla matrice delle features, che dalla matrice di adiacenza, eliminando poi anche le features degli archi che connettono gli atomi di idrogeno al resto del grafo). Penso però che il DataLoader di pytorch_geometric non carichi gli idrogeni. Potete controllare stampando la dimensione massima dei grafi che il DataLoader vi passa: se la dimensione massima è 29 ci sono gli idrogeni, se invece è 9 non ci sono.

  </details>

5. come faccio a far sì che la matrice adiacente e la matrice dei legami degli atomi siano correttamente matchate

6)  Per ottenere la generazione del vettore delle features dei nodi sia corretto aggiungere degli strati in più nel decoder della rete perchè di default il modello restituisce solo la matrice adiacente.
     <details>

    > Per quanto riguarda invece l'output, l'idea di aggiungere una parte che generi le features dei nodi mi sembra ottima. Potreste aggiungere un modulo anche piccolo che restituisca i vettori one-hot tramite softmax. Se poi il softmax non dovesse funzionare o dovesse comportarsi in modo troppo ripetitivo, potremo sostituirlo con un layer particolare che ha un'uscita stocastica (gumbel softmax).

     </details>

7)  non siamo ancora sicuri di come aggiungere le features degli edges, avevamo pensato prima di tutto a concatenare direttamente le features degli edges a quelle dei nodi poi abbiamo pensato di modificare la matrice di adiacenza aggiungendo le features degli edges in modo da ottenere un tensore 3d dove ad ogni edge corrisponde la codifica one-hot del legame della molecola.
     <details>

    > Ho controllato la repo, e devo dire che la gestione delle features degli archi c'è, ma è effettivamente un po' intricata: viene definita una matrice "s" (non esattamente un nome esplicativo per una variabile) ottenuta cross-correlando le features degli archi tra di loro. La matrice "s" viene utilizzata dal metodo self.mpm(), che ripete alcune iterazioni di "message passing convoluzionale" tra i nodi del grafo. Questa procedura permette di incorporare le features degli archi all'interno delle "features rielaborate" in uscita dal processo di message passing, che vengono poi compresse nel vettore latente. Il message passing è quello descritto in questo articolo: https://arxiv.org/abs/1704.01212

    > Secondo me conviene mantenere il metodo così com'è, almeno per adesso.

     </details>

8. Poi per quanto riguarda la loss abbiamo seguito il codice ovvero la somma della KL e la binary-cross-entropy tra la matrice adiacente vera e quella ricostruita dalla rete partendo dal vettore delle features dei nodi.
   E volevamo quindi sapere se era giusto lasciarla così oppure aggiungere un altro pezzo per considerare anche gli strati aggiunti in più nel decoder per ottenere il vettore di features dei nodi.

## appunti e domande 19/02:

- chiedere come calcolare la matrice di similarità tra le matrici di adiacenza e i vettori delle features degli edges

- chidere se è necessario modificare la loss dopo aver aggiunto i nuovi strati per le features dei nodi/edges

- problema del match tra la matrice adiacente e quella delle features degli edge: il numero di edges nella matrice adiacente è diverso da quello del edges features

- fare il one-hot encoding dell'uscita delle features dei nodi

  - aggiungere la softmax sull'uscita delle features dei nodi

- aggiungere latent diffusion:
  - quale loss utilizzare? quella del diffusion model o quella del graphVAE?

## Resurces

- Latent Diffusion:

  - [Latent Diffusion](https://github.com/CompVis/latent-diffusion)

    - [GITHUB - Notebook on ImageNet](https://github.com/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb)

  - [Hugging Face](https://huggingface.co/fusing/latent-diffusion-text2im-large)

  - [GITHUB REPO - Diffusers](https://github.com/huggingface/diffusers)

  - [List of papers and repo for Graph Diffusion](https://github.com/ChengyiLIU-cs/Generative-Diffusion-Models-on-Graphs)

  - [COLAB - Diffusion for MNIST](https://colab.research.google.com/github/JeongJiHeon/ScoreDiffusionModel/blob/main/DDPM/DDPM_MNIST.ipynb)

  - [Make Diffusion From scratch](https://tree.rocks/make-diffusion-model-from-scratch-easy-way-to-implement-quick-diffusion-model-e60d18fd0f2e)

- VGAE:

  - [Tutorial VGAE](https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129)
  - [Molecular Generation with QM9 dataset](https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan-graphs.py)
  - [Implementation of Small Molecular Generation with TensorFlow](https://github.com/poloarol/small-molecules/tree/main)
  - [Another Implementation but with ZINC dataset](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/batched_Molecule_Generation.py)
