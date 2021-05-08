import argparse
import os
import typing as t

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pydantic import ValidationError

from config import Config
from helpers import Logger
from helpers import LoggerMixin
from helpers import RuntimeArgsValidator


logger = Logger(__file__, verbose=Config.VERBOSE)


class TrainedClassificationModel(LoggerMixin):
    """
    Wrapper class for a trained classification model. Any pytorch model could
    be plugged in provided that it relies on the preprocessing steps
    implemented below
    """

    def __init__(
        self,
        model_name: str,
        model_weights: str,
        model_classes: str,
        device: str = "gpu",
    ) -> None:
        self._model_name = model_name
        try:
            self._model = torch.load(model_weights)
        except Exception as e:
            self.logger.exception(
                f"Failed to load the model {model_name}. Error: {e}"
            )
            raise e
        self._model.eval()
        self._classes = TrainedClassificationModel.read_classes(model_classes)
        self.logger.info(
            f"The model {model_name} initialized with classes {self._classes}"
        )
        if device == "gpu":
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        self._transform = self._get_transformation()

    def process_batch(self, batch: t.List[np.ndarray]) -> t.Any:
        to_classify: t.List[torch.Tensor] = []
        for image in batch:
            preprocessed_image = self._transform(Image.fromarray(image))
            to_classify.append(torch.unsqueeze(preprocessed_image, 0))
        torch_batch = torch.cat(to_classify)
        torch_batch = torch_batch.to(self._device)
        with torch.no_grad():
            output = self._model(torch_batch)
        return [
            self._classes[out.data.numpy().argmax()] for out in output.cpu()
        ]

    def _get_transformation(self) -> torchvision.transforms.Compose:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (Config.INPUT_SIZE, Config.INPUT_SIZE)
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
                ),
            ]
        )

    @staticmethod
    def read_classes(path_to_file: str) -> t.List[str]:
        with open(path_to_file, "r") as file:
            return [item.strip() for item in file.readlines()]


def get_image_path(folder: str) -> t.Iterator[str]:
    for item in os.listdir(folder):
        if not any(item.lower().endswith(ext) for ext in Config.ALLOWED_EXTS):
            logger.error(
                f"Cannot validate with the file {item}. Unsupported extension"
            )
            continue
        yield os.path.join(folder, item)


def get_batch(folder: str, batch_size: int) -> t.Iterator[t.List[np.ndarray]]:
    batch: t.List[np.ndarray] = []
    to_break = False
    next_image_gen = get_image_path(folder)
    while True:
        if len(batch) < batch_size:
            try:
                image_path = next(next_image_gen)
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(
                        f"Failed to open image: {image_path}. Skipped"
                    )
                    continue
                batch.append(image)
            except StopIteration:
                to_break = True
        if len(batch):
            yield batch
            batch = []
        if to_break:
            break


def validate_class(
    model: TrainedClassificationModel, folder: str, expected_class: str
) -> float:
    total_validation_images = len(os.listdir(folder))
    if not total_validation_images:
        raise Exception(f"The validation folder {folder} is empty")
    correct = 0
    for batch in get_batch(folder, Config.INFERENCE_BATCH):
        preds: t.List[str] = model.process_batch(batch)
        correct += preds.count(expected_class)
    return correct / total_validation_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_folder", type=str, required=True)
    parser.add_argument(
        "--model_weights", type=str, default="output/model_weights.pth"
    )
    parser.add_argument(
        "--model_classes", type=str, default="output/classes.txt"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        RuntimeArgsValidator(**(vars(args)))
    except ValidationError as e:
        logger.exception(f"Failed to validate the provided args. Error: {e}")
        raise e

    model = TrainedClassificationModel(
        "DigitsClassifier", args.model_weights, args.model_classes
    )
    validation_folder = args.validation_folder
    classes_to_validate = os.listdir(validation_folder)
    logger.info(
        f"The following classes will be validated (found in the folder "
        f"provided): {classes_to_validate}"
    )
    accuracies = 0.0
    for cls in classes_to_validate:
        logger.info(f"Validating class: {cls}")
        accuracy = validate_class(
            model, os.path.join(validation_folder, cls), str(cls)
        )
        logger.info(f"Validation results for class {cls} is {accuracy:.4f}")
        accuracies += accuracy

    accuracy = accuracies / len(classes_to_validate)
    logger.info(f"ACCURACY: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    main()
