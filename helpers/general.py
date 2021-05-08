import typing as t

import matplotlib.pyplot as plt


def visualise_training_results(
        acc_history: t.Sequence[float],
        loss_history: t.Sequence[float]
) -> None:
    plt.subplot(1, 2, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(acc_history, linewidth=3)

    plt.subplot(1, 2, 2)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(loss_history, linewidth=3)

    plt.show()
