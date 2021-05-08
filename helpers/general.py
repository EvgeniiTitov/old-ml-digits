import os
import typing as t

import matplotlib.pyplot as plt
from pydantic import BaseModel
from pydantic import validator


def visualise_training_results(
    acc_history: t.Sequence[float], loss_history: t.Sequence[float]
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


class RuntimeArgsValidator(BaseModel):
    validation_folder: str
    model_weights: str
    model_classes: str

    @validator("validation_folder")
    def check_folder_exists(cls, folder: str) -> str:
        if not os.path.exists(folder):
            raise FileNotFoundError("Failed to locate the validation folder")
        return folder

    @validator("model_weights")
    def check_model_weights(cls, weights: str) -> str:
        if not os.path.exists(weights):
            raise FileNotFoundError("Failed to locate the model weights")
        if not os.path.splitext(weights)[-1].lower() in [".pth", ".pt"]:
            raise Exception(
                "Incorrect weights. Expected a pytorch ext: .pth or .pt"
            )
        return weights

    @validator("model_classes")
    def check_model_classes(cls, classes_path: str) -> str:
        if not os.path.exists(classes_path):
            raise FileNotFoundError("Failed to locate the classes txt")
        if not os.path.splitext(classes_path)[-1].lower() in [".txt"]:
            raise Exception("Model classes must be a txt file")
        return classes_path
