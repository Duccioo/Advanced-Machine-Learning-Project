import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
import matplotlib.colors as mcolors
import numpy as np


def trasforma_tensore(tensore):
    # Se la lunghezza del tensore è inferiore a 3, aggiungi zeri per raggiungere una lunghezza di 3
    while len(tensore) < 3:
        tensore = list(tensore)
        tensore.append(0)

    # Prendi i primi 3 elementi del tensore e normalizzali tra 0 e 1
    tupla_risultato = np.array(tuple(x / max(tensore) for x in tensore[:3]))

    return tupla_risultato


# Caricamento del dataset QM9
dataset = QM9(root="data/QM9")

# Estrazione del primo grafo del dataset
data = dataset[0]
print(data)

print(data.edge_index)
print("**" * 10)
print(dataset[1])
print(dataset[1].x)
print(dataset[1].edge_index)
print("--" * 10)
print(dataset[2])
print(dataset[2].x)
print(dataset[2].edge_index)
print("--" * 10)
print(dataset[3])
print(dataset[3].x)
print(dataset[3].edge_index)

exit()

# Creazione del grafo NetworkX
G = nx.Graph()
for i in range(data.num_nodes):
    G.add_node(i, atom=data.x[i][0].item())

for j, k in zip(data.edge_index[0], data.edge_index[1]):
    G.add_edge(j.item(), k.item())

# Disegno del grafo
pos = nx.spring_layout(G)

tensor_hash_x = trasforma_tensore(data.x[:, 0])
tensor_hash_edge = trasforma_tensore(data.edge_index[:, 0])

rgba_node = mcolors.to_rgba(tensor_hash_x, alpha=None)
rgba_edge = mcolors.to_rgba(tensor_hash_edge, alpha=None)
# Conversione del valore di feature in una tupla RGBA
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=rgba_node,
    edge_color=rgba_edge,
    width=1,
    font_size=8,
    font_weight="bold",
)
node_labels = nx.get_node_attributes(G, "atom")
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=17)
plt.show()
