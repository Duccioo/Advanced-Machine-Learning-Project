import matplotlib.pyplot as plt
import numpy as np


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

    import time

    def Generator(n):
        for i in range(n):
            yield i
            
    prova =[1,2,3,4,5,6,7,8,9]

    for i in pit(range(2), color="red"):
        for j in pit(range(3), color="green"):
            for i, k in pit(enumerate(prova), total=9, color="blue"):
                    print(f"A O {k}")
               
                    time.sleep(0.05)
