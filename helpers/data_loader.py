import os
import typing as t

import torch
from torchvision import datasets
from torchvision import transforms

from .logger_mixin import LoggerMixin
from config import Config


class DatasetLoader(LoggerMixin):
    def __init__(
        self,
        path_to_dataset: str,
        augmentation: bool,
        input_size: int,
        batch_size: int,
    ) -> None:
        self._path_to_dataset = path_to_dataset
        self._augmentation = augmentation
        self._input_size = input_size
        self._batch_size = batch_size
        self.logger.info("DatasetLoader initialized")

    def _get_transformations(self) -> t.MutableMapping[str, t.Any]:
        if self._augmentation:
            self.logger.info("Using augmentation: Rotations, ColorJitter")
            data_transforms = {
                "train": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._input_size, self._input_size)
                        ),
                        transforms.RandomRotation(
                            degrees=Config.ROTATION_DEGREES
                        ),
                        transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
                        ),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._input_size, self._input_size)
                        ),
                        transforms.CenterCrop(self._input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
                        ),
                    ]
                ),
            }
        else:
            data_transforms = {
                "train": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._input_size, self._input_size)
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
                        ),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._input_size, self._input_size)
                        ),
                        transforms.CenterCrop(self._input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            Config.NORMALIZATION_MEAN, Config.NORMALIZATION_STD
                        ),
                    ]
                ),
            }
        return data_transforms

    def get_training_dataset(self) -> t.Tuple:
        data_transforms = self._get_transformations()
        self.logger.info("Image transformations generated")

        image_datasets = {
            phase: datasets.ImageFolder(
                os.path.join(self._path_to_dataset, phase),
                data_transforms[phase],
            )
            for phase in ["train", "val"]
        }
        self.logger.info("Image datasets created")

        data_loaders = {
            phase: torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=self._batch_size,
                shuffle=True,
            )
            for phase in ["train", "val"]
        }
        self.logger.info("Data loaders for both training and val created")

        dataset_sizes = {
            phase: len(image_datasets[phase]) for phase in ["train", "val"]
        }
        class_names = image_datasets["train"].classes
        self.logger.info(
            f"The following classes detected: {class_names}. "
            f"Dataset size: {dataset_sizes}"
        )
        return image_datasets, data_loaders, dataset_sizes, class_names
