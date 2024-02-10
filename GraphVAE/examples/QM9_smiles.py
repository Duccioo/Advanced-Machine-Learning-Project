from rdkit import Chem
from torch_geometric.datasets import QM9


# from __future__ import print_function
from rdkit import Chem

# from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import Draw

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import os
from rdkit.Chem import AllChem


def moltosvg(mol, molSize=(450, 150), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def render_svg(svg):
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace("svg:", ""))


def save_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


def save_svg(mol, filepath):
    svg = moltosvg(mol)
    with open(filepath, "w") as fw:
        fw.write(svg)


def data_to_smiles(data, atomic_numbers_idx: int = 5):
    # Crea un dizionario per mappare i numeri atomici ai simboli atomici
    atomic_numbers = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    # Crea un dizionario per mappare la rappresentazione one-hot encoding ai tipi di legami
    bond_types = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
    }

    # Crea un oggetto molecola vuoto
    mol = Chem.RWMol()

    # Aggiungi gli atomi alla molecola
    number_atom = 0
    for idx, atom in enumerate(data.x.tolist()):
        number_atom += 1
        # atom = mol.AddAtom(Chem.Atom(atomic_numbers[int(atom[atomic_numbers_idx])]))
        # print(data.x)
        # print(atomic_numbers[int(data.z[idx])])
        atom_ = Chem.Atom(atomic_numbers[int(data.z[idx])])
        # print(data.pos[idx, 0])
        atom_.SetDoubleProp("x", data.pos[idx, 0].item())
        atom_.SetDoubleProp("y", data.pos[idx, 1].item())
        atom_.SetDoubleProp("z", data.pos[idx, 2].item())
        mol.AddAtom(atom_)

    # Aggiungi i legami alla molecola
    edge_index = data.edge_index.tolist()
    bond_saved = []

    for idx, start_end in enumerate(zip(edge_index[0], edge_index[1])):
        start, end = start_end
        bond_type_one_hot = int((data.edge_attr[idx]).argmax())
        bond_type = bond_types[bond_type_one_hot]
        if (start, end, bond_type) not in bond_saved and (
            end,
            start,
            bond_type,
        ) not in bond_saved:
            # print("Bound saved, ", start, end, bond_type)
            bond_saved.append((start, end, bond_type))
            mol.AddBond(start, end, bond_type)

    # Converti la molecola in una stringa SMILES
    smiles = Chem.MolToSmiles(mol)
    return smiles, number_atom


def main():
    # Carica il dataset QM9
    # dataset = QM9(root="data/QM9_dense", transform=T.ToDense()) <- così ho già lo smiles!!
    dataset = QM9(root="data/QM9")
    print(dataset[0], "\n\n")

    # Prendi la prima molecola nel dataset
    index = range(1, 24)
    error = 0
    for elem in index:
        data = dataset[elem]
        # print(data.smiles)

        sm, number = data_to_smiles(data)
        # print(sm)
        dirpath = ""
        if sm != data.smiles:
            error = error + 1
            print(
                f"ohoh, problema in {elem}! ->",
                data.smiles,
                "<00000>",
                sm,
                "$$$$$$",
                number,
            )
            print(data)
            print(data.x)
            print(data.edge_index)
            print(data.edge_attr)

        # save_filepath = os.path.join(dirpath, "mol_{}.png".format(elem))
        # mol = Chem.MolFromSmiles(sm)
        # mol = Chem.AddHs(mol)
        # save_png(mol, save_filepath, size=(600, 600))

        # save_filepath = os.path.join(dirpath, "mol_{}_2.png".format(elem))
        # mol = Chem.MolFromSmiles(data.smiles)
        # mol = Chem.AddHs(mol)
        # save_png(mol, save_filepath, size=(600, 600))

    print(
        f"\nHo trovato esattamente {error} discrepanze tra lo smiles e la molecola matriciale"
    )


if __name__ == "__main__":
    main()
