import os
import json
from datetime import datetime

from rich import box
from rich.console import Console
from rich.table import Table
from rdkit.Chem import Draw
from rdkit import Chem


def console_bullet_list(console, list_elem: list = []):
    if isinstance(list_elem, list):
        for element in list_elem:
            console.print("- ", element)


def console_table_dict(console, dict: dict = {}, header: tuple = ("KEY", "VALUE"), inplace: bool = True):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    table.add_column(header[0])
    table.add_column(header[1])
    for key, value in dict.items():
        table.add_row(key, str(value))

    if inplace:
        console.print(table)
    return table


def console_matrix(console, matrix: list = [], inplace: bool = True):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    num_col = len(matrix[0])
    for i in range(num_col):
        table.add_column(str(i + 1))
    for row in matrix:
        row = [str(i) for i in row]
        table.add_row(*row)
    if inplace:
        console.print(table)
    return table


class Summary:
    def __init__(self, directory="model_saved_prova", model_type="GraphVAE_BASE"):
        self.directory_base = directory
        self.model_type = model_type

        if os.path.exists(self.directory_base):
            if any(file.endswith("FINAL.pth") for file in os.listdir(self.directory_base)):
                self.directory_base += "_1"
                os.makedirs(self.directory_base)
        else:
            os.makedirs(self.directory_base)

        self.directory_checkpoint = os.path.join(self.directory_base, "checkpoints")
        if not os.path.exists(self.directory_checkpoint):
            os.makedirs(self.directory_checkpoint)
        self.directory_log = os.path.join(self.directory_base, "logs")
        if not os.path.exists(self.directory_log):
            os.makedirs(self.directory_log)

    def save_model_json(self, model_params: list = [], file_name="hyperparams.json"):
        # salvo gli iperparametri:
        json_path = os.path.join(self.directory_base, file_name)
        # Caricamento dei dati dal file JSON
        with open(json_path, "w") as file:
            json.dump(model_params, file)

    def save_summary_training(
        self, dataset_params: list[dict] = [], model_params: list[dict] = [], example_data: dict = {}
    ):
        now = datetime.now().strftime("%Y-%m-%d %H-%M")
        filename = os.path.join(self.directory_base, "summary_experiment.md")
        with open(filename, "w") as file:
            console = Console(file=file)
            console.print("# Experiment Details")

            console.print(f"> from experiment with {self.model_type}")
            console.print(f"> on {now}")
            console.print("## Model")
            try:
                console_table_dict(console, model_params[0])
            except:
                console_table_dict(console, model_params)
            console.print("## Dataset")
            console.print(
                "- Dataset used [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html)"
            )
            console_table_dict(console, dataset_params[0])

            self.save_example_data(console, example_data)

    def save_example_data(self, console: Console, example_data, details=True):
        console.print("## Example data")
        summary = "Example Data from QM9 dataset Padded"
        console.print(f"<details><summary>{summary}</summary>\n")
        for key, value in example_data.items():

            if isinstance(value, int) or isinstance(value, str):
                console.print(f"#### {key} :\n - {str(value)}")
                if key == "smiles":
                    mol = Chem.MolFromSmiles(str(value))
                    mol_filepath = os.path.join(self.directory_base, "example_molecule.png")
                    Draw.MolToFile(mol, mol_filepath)
                    console.print("\n<img src='example_molecule.png'>")

            else:
                console.print("#### " + key + " :")
                console.print("> __SHAPE__ : " + str(value.shape))
                # print(value.tolist())
                table = console_matrix(console, value.tolist())
        console.print("</details>")


if __name__ == "__main__":
    with open("prova.md", "w") as file:
        console = Console(file=file)

        table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
        table.add_column("Released", justify="center", style="cyan", no_wrap=True)
        table.add_column("Title", justify="center", style="magenta")

        table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690")
        table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
        table.add_row("Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889")
        for i in range(3):
            table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", f"{i*100}")

        console.print("## INFORMATION", style="blue")
        console.print(table)
        console.print("## DANGER!", style="red on white")
        console.print_json(data={"prova": "proav1"})
        lista = ["prova1", "prova2", "prova3"]
        console_bullet_list(console, lista)

        dicta = {"prova1": "prova1", "prova2": "prova2", "prova3": "prova3"}
        console_table_dict(console, dicta)

        matrix = [["ppp", "bbbb", "zzzz"], ["ppp", "bbbb"], ["ppp", "bbbb"]]
        prova = np.ones((3, 7))
        console_matrix(console, prova)
