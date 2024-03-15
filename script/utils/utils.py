import os
import random
import matplotlib.pyplot as plt
import numpy as np
import hashlib


import torch
from datetime import datetime
import rdkit.Chem as Chem
from rdkit import rdBase

blocker = rdBase.BlockLogs()

# ---
_checkpoint_base_name = "checkpoint_"


def generate_unique_id(params: list = [], lenght: int = 10) -> str:
    input_str = ""

    # Concateniamo le stringhe dei dati di input
    for param in params:
        if type(param) is list:
            param_1 = [str(p) if not callable(p) else p.__name__ for p in param]
        else:
            param_1 = str(param)
        input_str += str(param_1)

    # Calcoliamo il valore hash SHA-256 della stringa dei dati di input
    hash_obj = hashlib.sha256(input_str.encode())
    hex_dig = hash_obj.hexdigest()

    # Restituiamo i primi 8 caratteri del valore hash come ID univoco
    return hex_dig[:lenght]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def check_base_dir(*args):
    """
    A function to check and create a directory based on the given arguments.

    Parameters:
    *args: variable number of arguments representing the path components

    Returns:
    str: the full path after checking and creating the directory
    """
    # take the full path of the folder
    absolute_path = os.path.dirname(__file__)

    full_path = absolute_path
    # args = [item for sublist in args for item in sublist]
    for idx, path in enumerate(args):
        # check if arguments are a list

        if type(path) is list:
            # path = [item for sublist in path for item in sublist if not isinstance(item, str)]
            for micro_path in path:
                if isinstance(micro_path, list):
                    for micro_micro_path in micro_path:
                        full_path = os.path.join(full_path, micro_micro_path)
                else:
                    full_path = os.path.join(full_path, micro_path)

        else:
            full_path = os.path.join(full_path, path)

        # check the path exists
        if not os.path.exists(full_path):
            # if doesn't, create it:
            os.makedirs(full_path)

    return full_path


def graph_to_mol(adj, node_labels, edge_features, sanitize, cleanup):
    mol = Chem.RWMol()
    smiles = ""

    node_labels = node_labels[:, 5:9]
    node_labels = torch.argmax(node_labels, dim=1)

    atomic_numbers = {0: "C", 1: "N", 2: "O", 3: "F"}

    # Crea un dizionario per mappare la rappresentazione one-hot encoding ai tipi di legami
    bond_types = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
    }
    # print(f"Node Labels {node_labels}")

    idx = 0

    for node_label in node_labels.tolist():
        # print(f"Adding atom {atomic_numbers[node_label]}")
        mol.AddAtom(Chem.Atom(atomic_numbers[node_label]))

    for edge in np.nonzero(adj).tolist():
        start, end = edge[0], edge[1]
        if start > end:

            bond_type_one_hot = int((edge_features[idx]).argmax())
            bond_type = bond_types[bond_type_one_hot]
            # print(f"ADDING BOUND {bond_type} to {start} and {end}")
            idx += 1

            try:
                mol.AddBond(int(start), int(end), bond_type)
            except:
                print("ERROR Impossibile aggiungere legame")
    if sanitize:
        try:
            flag = Chem.SanitizeMol(mol, catchErrors=True)
            # Let's be strict. If sanitization fails, return None
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                mol = None

                # print("Sanitize Failed")

        except Exception:
            # print("Sanitize Failed")
            mol = None

    if cleanup and mol is not None:
        try:
            mol = Chem.AddHs(mol, explicitOnly=True)
        except:
            pass

        try:
            smiles = Chem.MolToSmiles(mol)
            smiles = max(smiles.split("."), key=len)
            if "*" not in smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                print("mol from smiles failed")
                mol = None
        except:
            # print("error generic")
            smiles = Chem.MolToSmiles(mol)

            mol = None

    return mol, smiles


# ------------------------------------SAVING & LOADING-----------------------------------------
def load_from_checkpoint(checkpoint_path, model, optimizer=None, schedule=None):
    """
    Load model from checkpoint, return the step, epoch and model step saved in the file.

    Parameters:
        - checkpoint_path (str): path to checkpoint
        - model (nn.Module): model to load
        - optimizer(torch.optim): optimizer for model
        - schedule
    """
    data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(data["model"])

    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])

    if schedule is not None:
        schedule.load_state_dict(data["schedule"])

    try:
        data.get("loss")

        element = (data["step"], data["epoch"], data["loss"], data["other"])
    except:
        element = (data["step"], data["epoch"])

    return data


def save_checkpoint(
    checkpoint_path,
    checkpoint_name,
    model,
    step,
    epoch,
    loss: list = None,
    optimizer=None,
    schedule=None,
    other=None,
):
    """
    Save a checkpoint for the model's state, including step, epoch, loss, optimizer, and schedule information.
        Parameters:
            checkpoint_path (str): The path to save the checkpoint.
            checkpoint_name (str): The name of the checkpoint file.
            model (nn.Module): The model for which the checkpoint is being saved.
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            loss (list, optional): The loss information to be included in the checkpoint. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): The optimizer's state to be included in the checkpoint. Defaults to None.
            schedule (torch.optim.lr_scheduler._LRScheduler, optional): The schedule's state to be included in the checkpoint. Defaults to None.
        Returns:
            None

    """
    state_dict = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
    }
    if loss is not None:
        state_dict.update(
            {
                "loss": loss,
            }
        )

    if optimizer is not None:
        state_dict.update(
            {
                "optimizer": optimizer.state_dict(),
            }
        )
    if schedule is not None:
        state_dict.update(
            {
                "schedule": schedule.state_dict(),
            }
        )
    if other is not None:
        state_dict.update(
            {
                "other": other,
            }
        )

    if os.path.exists(checkpoint_path):
        torch.save(state_dict, os.path.join(checkpoint_path, checkpoint_name))

    else:
        os.mkdir(checkpoint_path)
        torch.save(state_dict, os.path.join(checkpoint_path, checkpoint_name))


def latest_checkpoint(root_dir, base_name):
    """
    Find the latest checkpoint in a directory.
    Parameters:
        - root_dir (str): root directory
        - base_name (str): base name of checkpoint
    """
    check_base_dir(root_dir)

    checkpoints = [chkpt for chkpt in os.listdir(root_dir) if base_name in chkpt]
    if len(checkpoints) == 0:
        return None
    latest_chkpt = ""
    latest_step = -1
    latest_epoch = -1
    for chkpt in checkpoints:
        step = torch.load(os.path.join(root_dir, chkpt))["step"]
        epoch = torch.load(os.path.join(root_dir, chkpt))["epoch"]
        if step > latest_step or (epoch > latest_epoch):
            latest_epoch = epoch
            latest_chkpt = chkpt
            latest_step = step
    return os.path.join(root_dir, latest_chkpt)


def clean_old_checkpoint(folder_path, percentage):
    """
    Delete old checkpoints from a directory with a given percentage.
    Parameters:
        - folder_path (str): path to directory where checkpoints are stored
        - percentage (float): percentage of checkpoints to delete

    """

    # ordina i file per data di creazione
    files = [(f, os.path.getctime(os.path.join(folder_path, f))) for f in os.listdir(folder_path)]
    files.sort(key=lambda x: x[1])

    # calcola il numero di file da eliminare
    files_to_delete = int(len(files) * percentage)

    # cicla attraverso i file nella cartella
    for file_name, creation_time in files[:-files_to_delete]:
        # costruisce il percorso completo del file
        file_path = os.path.join(folder_path, file_name)
        # elimina il file
        if file_name.endswith(".pt"):
            os.remove(file_path)


# ------------------------------------LOGGING------------------------------------


def log(
    out_dir,
    model,
    discriminator,
    optimizer_G,
    params,
    g_step,
    checkpoints_dir="",
    d_loss=0,
    run_epoch=0,
    run_step=0,
    checkpoint_base_name=_checkpoint_base_name,
    accuracy=0,
):
    if run_step % params.steps_per_img_save == 0 and run_step > 0:

        # controllo se il parametro di telegram è stato impostato
        if params.telegram == True:
            # se è attivo invio un messaggio dal bot
            telegram_alert.alert(
                "TRAINING",
                run_step,
                params.steps,
                f"Loss: <b>{(d_loss.item()):.2f}</b>\nEpoca: <b>{(run_epoch)}</b>\tSTEP: <b>{(run_step)}</b>",
            )

    if run_step == params.steps:
        # quando finisco il training
        save_checkpoint(
            checkpoints_dir,
            f"model_final_{run_step}_{run_epoch}.pt",
            model,
            discriminator,
            step=run_step,
            epoch=run_epoch,
            g_step=g_step,
        )

    if run_step % params.steps_per_checkpoint == 0 and run_step > 0:
        # ogni tot mi salvo il checkpoint anche degli optimizer

        save_checkpoint(
            checkpoints_dir,
            f"{checkpoint_base_name}_s{run_step}_e{run_epoch}.pt",
            model,
            step=run_step,
            epoch=run_epoch,
            optimizer=optimizer_G,
        )

    if run_step % params.steps_per_val == 0 and run_step > 0:
        # save validation
        save_validation(
            accuracy,
            run_step,
            run_epoch,
            d_loss,
            filename=os.path.join(out_dir, "validation.csv"),
        )

    if run_step % params.steps_per_clean == 0 and run_step > 0:
        # cancello un po di checkpoint se sono troppo vecchi
        clean_old_checkpoint(checkpoints_dir, 0.5)


def log_metrics(
    epochs: int,
    train_loss: list,
    train_accuracy: list = [],
    val_accuracy: list = [],
    date: list = [],
    total_batch: int = None,
    log_file_path: str = None,
    plot_file_path: str = None,
    metric_name: str = "Accuracy",
    title: str = "Training and Validation Metrics",
    plot_show: bool = False,
    plot_save: bool = False,
):
    """
    Log loss and accuracy metrics at each epoch and create a plot.
    By default the metrics are saved in the "logs" folder in the current directory.
    Args:
        epochs (int): Total number of training epochs.
        train_loss (list): List of training loss values for each epoch.
        train_accuracy (list): List of training accuracy values for each epoch.
        val_accuracy (list): List of validation accuracy values for each epoch.
        date (list): List of dates for each epoch.
        log_file_path (str): Path to the log file where metrics will be saved.
        plot_file_path (str): Path to save the plot image.
        metric_name (str): Name of the metric (default is "Accuracy").
        title (str): Title of the plot (default is "Training and Validation Metrics").
        plot_show (bool): Whether to show the plot (default is False).
        plot_save (bool): Whether to save the plot (default is False).
    Returns:
        None
    """

    # plt.figure(figsize=(10, 6))
    plt.figure()

    if total_batch:

        num_batch_totali = epochs * total_batch

        # Crea una lista di numeri di batch
        total_elemets = list(range(1, num_batch_totali + 1))
        # Aggiungi le barre verticali per le epoche
        list_batch_epoch = []
        for epoca in range(1, epochs + 2):
            batch_inizio_epoca = (epoca - 1) * total_batch

            plt.axvline(x=batch_inizio_epoca, color="red", linestyle="--", alpha=0.3)
            list_batch_epoch.append(batch_inizio_epoca)

        plt.xticks(list_batch_epoch, [f"{epoca}" for epoca in range(0, epochs + 1)])

    else:
        total_elemets = np.arange(1, epochs + 1)

    plt.plot(total_elemets, train_loss, label="Train Loss", marker="o")

    metric_label = "Loss"
    if val_accuracy or train_accuracy:
        metric_label += f", {metric_name}"

    if not date:
        date = [datetime.now() for _ in total_elemets]

    if not train_accuracy:
        train_accuracy = [0] * len(total_elemets)
    else:
        plt.plot(
            np.arange(1, len(total_elemets) + 1),
            train_accuracy,
            label=f"Train {metric_name}",
            marker="s",
        )
    if not val_accuracy:
        val_accuracy = [0] * len(total_elemets)
    else:
        plt.plot(
            total_elemets,
            val_accuracy,
            label=f"Validation {metric_name}",
            marker="^",
        )

    if log_file_path is None:
        print("Log file not found. Log will be created by default.")
        global_path = check_base_dir("..", "logs")
        today = datetime.now().strftime("%m-%d_%H-%M")
        log_file_path = os.path.join(global_path, f"metrics__{today}.csv")
        print(f"Log file path: {log_file_path}")

    with open(log_file_path, "w", newline="") as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Accuracy\n")
        for idx, elem in enumerate(total_elemets):
            log_file.write(
                f"{date[idx]}\t,{idx}\t,{train_loss[idx]:.4f},\t{train_accuracy[idx]:.4f},\t{val_accuracy[idx]:.4f}\n"
            )
    # Create the plot
    if plot_file_path is None and plot_save:
        print("Plot file not found. Plot will not be created.")
        global_path = check_base_dir("..", "logs")
        today = datetime.now().strftime("%m-%d_%H-%M")
        plot_file_path = os.path.join(global_path, f"plot__{today}.png")

    plt.xlabel("Epochs")
    plt.ylabel(metric_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if plot_save:
        plt.savefig(plot_file_path)
    if plot_show:
        plt.show()


if __name__ == "__main__":

    # Example usage:
    epochs = 10
    train_loss = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07] * epochs
    train_accuracy = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95]
    val_accuracy = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91]
    # e che conosci il numero totale di epoche e il numero di elementi per batch
    elementi_per_batch = 10

    log_metrics(epochs, train_loss, plot_show=True, elements_per_batch=elementi_per_batch)
