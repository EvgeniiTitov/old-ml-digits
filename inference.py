import typing as t
import os
import argparse

import torch
import cv2
import numpy as np
import torchvision
from PIL import Image

from helpers import LoggerMixin, Logger
from config import Config


logger = Logger(__file__, verbose=Config.VERBOSE)


class TrainedClassifierModel(LoggerMixin):
    def __init__(
            self,
            model_name: str,
            model_weights: str,
            model_classes: str,
            device: str = "gpu"
    ) -> None:
        self._model_name = model_name
        try:
            self._model = torch.load(model_weights)
        except Exception as e:
            self.logger.exception(
                f"Failed to load the model. Error: {e}"
            )
            raise e
        self._model.eval()
        self._classes = TrainedClassifierModel.read_classes(model_classes)
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
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (Config.INPUT_SIZE, Config.INPUT_SIZE)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
            ),
        ])

    @staticmethod
    def read_classes(path_to_file: str) -> t.List[str]:
        with open(path_to_file, "r") as file:
            return [item.strip() for item in file.readlines()]


def get_batch(folder_path: str) -> t.List[np.ndarray]:
    batch = []
    for item in os.listdir(folder_path):
        if not any(item.lower().endswith(ext) for ext in Config.ALLOWED_EXTS):
            logger.error(
                f"Cannot validate of file {item}. Unsupported extension"
            )
            continue
        if len(batch) < Config.INFERENCE_BATCH:
            image_path = os.path.join(folder_path, item)
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to open image: {image_path}. Skipped")
                continue
            batch.append(image)
            continue
        if len(batch):
            yield batch
            batch = []
    if len(batch):
        yield batch


def validate_class(
        model: TrainedClassifierModel,
        folder: str,
        expected_class: str
) -> t.Tuple[int, int]:
    total_validation_images = len(os.listdir(folder))
    correct = 0
    for batch in get_batch(folder):
        preds: t.List[str] = model.process_batch(batch)
        correct += preds.count(expected_class)
    return total_validation_images, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_folder", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.validation_folder):
        raise FileNotFoundError("Failed to locate the validation folder")

    model = TrainedClassifierModel(
        "DigitsClassifier",
        os.path.join("output", "model_weights.pth"),
        os.path.join("output", "classes.txt")
    )
    validation_folder = args.validation_folder
    classes_to_validate = os.listdir(validation_folder)
    logger.info(
        f"The following classes will be validated: {classes_to_validate}"
    )

    total_global, correct_global = 0, 0
    for cls in classes_to_validate:
        logger.info(f"Validating class: {cls}")
        total, correct = validate_class(
            model, os.path.join(validation_folder, cls), str(cls)
        )
        logger.info(f"Validation results for class {cls}: {correct} / {total}")
        total_global += total
        correct_global += correct

    accuracy = correct_global / total_global
    logger.info(f"ACCURACY: {accuracy:.4f}")
    return 0


if __name__ == '__main__':
    main()
