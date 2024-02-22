import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary
from datetime import datetime
import json

# ---
import telegram_alert

_checkpoint_base_name = "checkpoint_"


def check_base_dir(*args):
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
            os.makedirs(full_path)

    return full_path


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

    return (data["step"], data["epoch"])


def save_checkpoint(
    checkpoint_path, checkpoint_name, model, step, epoch, optimizer=None, schedule=None
):
    """
    Save model to a checkpoint file.
    Parameters:
        - checkpoint_path (str): path to checkpoint
        - checkpoint_name (str): name of checkpoint
        - model (nn.Module): model model
        - discriminator (str): discriminator
        - step (int): current step
        - epoch (int): current epoch
        - optimizer (torch.optim): optimizer for model
    """

    state_dict = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
    }
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
    files = [
        (f, os.path.getctime(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
    ]
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


def save_model(
    path,
    name,
    model,
    discriminator,
    batch_size,
    model_size,
    discriminator_size,
):
    """
    Using summary function from torchinfo save the model architecture in a json file.

    Parameters:
        - path: path to save the model architecture
        - name: name of the file to save
        - model: model model
        - discriminator: discriminator model
        - batch_size: batch size, parameter for function summary
        - model_size: model size, parameter for function summary
        - discriminator_size: discriminator size, parameter for function summary
    """

    # create new dictionary
    models = {"model": {}, "discriminator": {}}

    # prima mi carico il modele
    gen_sum = summary(
        model,
        input_size=(batch_size, model_size),
        verbose=0,
        col_names=["output_size"],
    )
    gen_arch = str(gen_sum).split(
        "================================================================="
    )[2]

    gen_size = str(gen_sum).split(
        "================================================================="
    )[4]

    for line in gen_arch.split("├─"):
        line.split("                  ")
        models["model"][line.split("                  ")[0].strip()] = line.split(
            "                  "
        )[1].strip()

    for line in gen_size.split("\n"):
        try:
            models["model"][line.split(":")[0]] = float(line.split(":")[1])
        except:
            pass

    # passo al discriminatore
    disc_sum = summary(
        discriminator,
        input_size=((batch_size,) + discriminator_size),
        verbose=0,
        col_names=["output_size"],
    )
    disc_arch = str(disc_sum).split(
        "================================================================="
    )[2]

    disc_size = str(disc_sum).split(
        "================================================================="
    )[4]

    for line in disc_arch.split("├─"):
        line.split("                  ")
        models["discriminator"][line.split("                  ")[0].strip()] = (
            line.split("                  ")[1].strip()
        )

    for line in disc_size.split("\n"):
        try:
            models["discriminator"][line.split(":")[0]] = float(line.split(":")[1])
        except:
            pass

    if os.path.isdir(path) == False:
        os.mkdir(path)

    with open(os.path.join(path, name + ".json"), "w") as f:
        json.dump(models, f)


def save_validation(
    accuracy,
    ssim,
    psnr,
    swd,
    recall,
    precision,
    f1_score,
    support,
    step=0,
    epoch=0,
    loss_d=0,
    loss_g=0,
    filename="metrics.csv",
):
    now = datetime.now()
    if loss_d != 0:
        loss_d_s = loss_d.item()
    else:
        loss_d_s = loss_d

    if loss_g != 0:
        loss_g_s = loss_g.item()

    else:
        loss_g_s = loss_g

    metrics = [
        step,
        epoch,
        f"{loss_d_s:.4f}",
        f"{loss_g_s:.4f}",
        f"{ssim:.4f}",
        f"{psnr:.4f}",
        f"{swd:.4f}",
        f"{accuracy:.4f}",
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1_score:.4f}",
        f"{support:.4f}",
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ]

    header = [
        "Step",
        "Epoch",
        "Loss D",
        "Loss G",
        "SSIM",
        "PSNR",
        "SWD",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_score",
        "Support",
        "TIME",
    ]
    # if os.path.exists(filename) == False:
    #     with open(filename, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerow(metrics)
    # else:
    #     with open(filename, "a") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(metrics)


# ------------------------------------LOGGING------------------------------------


def log(
    out_dir,
    model,
    discriminator,
    optimizer_G,
    params,
    g_step,
    progress_bar,
    checkpoints_dir="",
    imgs_dir="output",
    d_loss=0,
    run_epoch=0,
    run_step=0,
    checkpoint_base_name=_checkpoint_base_name,
    accuracy=0,
    device="cpu",
):
    if run_step % params.steps_per_img_save == 0 and run_step > 0:
        # ogni tot mi salvo anche un immagine
        # img_path = save_gen_img(
        #     model=model,
        #     noise=generate_noise(3, 100, 3, seed=1599, device=device),
        #     path=imgs_dir,
        #     title=(
        #         f"gen_s{str(run_step)}_e{(run_epoch)}_gl{(g_loss.item()):.2f}_dl{(d_loss.item()):.2f}.jpg"
        #     ),
        # )
        img_path = ""

        # controllo se il parametro di telegram è stato impostato
        if params.telegram == True:
            # se è attivo invio un messaggio dal bot
            telegram_alert.alert(
                "TRAINING",
                img_path,
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


def log_and_plot_metrics(
    epochs,
    train_loss,
    train_accuracy,
    val_accuracy,
    log_file_path,
    plot_file_path,
    show=False,
):
    """
    Log loss and accuracy metrics at each epoch and create a plot.

    Args:
        epochs (int): Total number of training epochs.
        train_loss (list): List of training loss values for each epoch.
        train_accuracy (list): List of training accuracy values for each epoch.
        val_accuracy (list): List of validation accuracy values for each epoch.
        log_file_path (str): Path to the log file where metrics will be saved.
        plot_file_path (str): Path to save the plot image.

    Returns:
        None
    """
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Accuracy\n")
        for epoch in range(epochs):
            log_file.write(
                f"{epoch+1}\t{train_loss[epoch]:.4f}\t{train_accuracy[epoch]:.4f}\t{val_accuracy[epoch]:.4f}\n"
            )

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), train_loss, label="Train Loss", marker="o")
    plt.plot(
        np.arange(1, epochs + 1), train_accuracy, label="Train Accuracy", marker="s"
    )
    plt.plot(
        np.arange(1, epochs + 1), val_accuracy, label="Validation Accuracy", marker="^"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file_path)
    if show:
        plt.show()


def pit(it, *pargs, **nargs):
    import enlighten

    global __pit_man__
    try:
        __pit_man__
    except NameError:
        __pit_man__ = enlighten.get_manager()
    man = __pit_man__
    try:
        it_len = len(it)
    except:
        it_len = None
    try:
        ctr = None
        for i, e in enumerate(it):
            if i == 0:
                ctr = man.counter(
                    *pargs, **{**dict(leave=False, total=it_len), **nargs}
                )
            yield e
            ctr.update()
    finally:
        if ctr is not None:
            ctr.close()


if __name__ == "__main__":

    # Example usage:
    epochs = 10
    train_loss = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07]
    train_accuracy = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95]
    val_accuracy = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91]
    log_file_path = "training_metrics.txt"
    plot_file_path = "training_metrics_plot.png"

    log_and_plot_metrics(
        epochs,
        train_loss,
        train_accuracy,
        val_accuracy,
        log_file_path,
        plot_file_path,
        show=False,
    )
