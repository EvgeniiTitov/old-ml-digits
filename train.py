import argparse
import copy
import os
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import ValidationError
from torchvision import models

from config import Config
from helpers import DatasetLoader
from helpers import Logger
from helpers import timer
from helpers import visualise_training_results


logger = Logger(__name__, verbose=Config.VERBOSE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_dataset",
        type=str,
        required=True,
        help="Path to the dataset in the ! ImageFolder ! format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to a folder to save training results",
    )
    return parser.parse_args()


def get_sota_model(model_name: str, pretrained: bool):
    """Creates an instance of a SOTA model to train"""
    try:
        model = getattr(models, model_name)(pretrained=pretrained)
    except AttributeError as e:
        logger.exception(f"Incorrect SOTA model name provided! Error: {e}")
        raise e
    return model


def freeze_layers(model):
    """Freezes all model's layers meaning they are not trainable - the gradient
    will not be calculated for those neurons, so they wont get "trained"
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def get_parameters_to_train(model, fine_tuning: bool) -> list:
    """Returns a list of trainable parameters - the ones for which the
    gradients will be calculated during backprop
    """
    trainable_parameters = model.parameters()
    if not fine_tuning:
        trainable_parameters = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                trainable_parameters.append(parameter)
    return trainable_parameters


def reshape_models_head(model, number_of_classes: int):
    """Reshapes the last dense layer(s) of the model to the number of classes
    the model will be trained for
    """
    if model.__class__.__name__ == "ResNet":
        number_of_filters = model.fc.in_features
        model.fc = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "AlexNet":
        # 6th Dense layer's input size: 4096
        number_of_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "VGG":
        # For both VGGs 16-19 classifiers are the same
        number_of_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "DenseNet":
        number_of_filters = model.classifier.in_features
        model.classifier = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "SqueezeNet":
        model.classifier[1] = nn.Conv2d(
            512, number_of_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = number_of_classes
    return model


@timer
def train_model(
    model,
    epochs: int,
    dataloaders: t.Mapping[str, torch.utils.data.DataLoader],
    device,
    optimizer: t.Union[optim.SGD, optim.Adam],
    loss_function: nn.CrossEntropyLoss,
    dataset_size: t.Mapping[str, int],
    scheduler: t.Optional[optim.lr_scheduler.StepLR] = None,
) -> tuple:
    val_accuracy_history, val_loss_history = [], []
    best_val_accuracy = 0
    best_accuracy_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch} / {epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()  # no gradients will be calculated, no backprop

            running_loss, running_corrects = 0.0, 0
            for batch, labels in dataloaders[phase]:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Gradients and backprop with subsequent parameter changes are
                # only allowed during the training phase
                with torch.set_grad_enabled(phase == "train"):
                    activations = model(batch)
                    loss = loss_function(activations, labels)

                    _, class_preds = torch.max(activations, dim=1)
                    if phase == "train":
                        loss.backward()  # calculate gradients
                        optimizer.step()  # tweak the parameters as per loss
                running_loss += loss.item() * batch.size(0)
                running_corrects += torch.sum(class_preds == labels.data)

            if phase == "train" and scheduler:
                scheduler.step()  # Slow down the learning rate

            epoch_loss = running_loss / dataset_size[phase]
            epoch_accuracy = (
                running_corrects.double() / dataset_size[phase]  # type: ignore
            )
            logger.info(
                f"{phase.upper()} Loss: {epoch_loss:.4f}. "
                f"Accuracy: {epoch_accuracy:.4f}"
            )
            if phase == "val":
                val_accuracy_history.append(epoch_accuracy.item())
                val_loss_history.append(epoch_loss)

            # Save the best weights
            if phase == "val" and epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                best_accuracy_epoch = epoch
    # Load the best model and return it along  side the training metrics
    model.load_state_dict(best_model_weights)
    logger.info("Loaded the best weights. Returning the model")

    return (
        model,
        val_accuracy_history,
        val_loss_history,
        best_val_accuracy,
        best_accuracy_epoch,
    )


def main() -> int:
    args = parse_args()
    # TODO: Validate arguments, both provided and from Config
    try:
        pass
    except ValidationError as e:
        logger.exception(
            f"Failed to validate arguments for training. Error: {e}"
        )
        raise e
    else:
        logger.info("Arguments parsed and validated")

    # Train the model
    dataset_manager = DatasetLoader(
        path_to_dataset=args.path_to_dataset,
        augmentation=Config.DO_AUGMENTATION,
        input_size=Config.INPUT_SIZE,
        batch_size=Config.BATCH_SIZE,
    )
    (
        image_dataset,
        data_loaders,
        dataset_sizes,
        class_names,
    ) = dataset_manager.get_training_dataset()
    logger.info("Dataloaders obtained")

    # Initialize the SOTA model
    model_to_train = get_sota_model(
        Config.MODEL_NAME, pretrained=Config.PRETRAINED
    )
    logger.info(f"Initialized the {Config.MODEL_NAME} model for training")

    # If no fine tuning required, freeze models layers to train only the head
    if not Config.FINE_TUNING:
        model_to_train = freeze_layers(model_to_train)
        logger.info("Model layers frozen as no fine tuning required")
    else:
        logger.info("No model layers have been frozen - doing fine tuning")

    # Reshape the model's head with a new one having the number of output
    # neurons equal to the number of classes we want to classify: 0 - 9
    if len(class_names) != 1000:
        model_to_train = reshape_models_head(model_to_train, len(class_names))
        logger.info(
            "Model's head has been reshaped to match the number of classes"
        )

    # Get a list of parameters to train
    trainable_parameters = get_parameters_to_train(
        model_to_train, Config.FINE_TUNING
    )
    logger.info("Got a list of trainable parameters")

    # Init an optimizer passing it the trainable parameters, learning rate and
    # some other hyperparameters
    if Config.OPTIMIZER.upper() == "ADAM":
        optimizer = optim.Adam(
            params=trainable_parameters, lr=Config.LR, betas=(0.9, 0.999)
        )
    elif Config.OPTIMIZER.upper() == "SGD":
        optimizer = optim.SGD(
            params=trainable_parameters, lr=Config.LR, momentum=0.9
        )
    else:
        raise NotImplementedError(
            "Requested optimizer is not implemented. Available: Adam, SGD"
        )
    logger.info("Optimizer initialized")

    # Initialize the loss function - the measure of how off our predictions
    # during training are compared to the desired result
    loss_function = nn.CrossEntropyLoss()
    logger.info("Loss function initialized")

    # Initialize a scheduler if requested - LR decay to prevent overfitting
    # by decreasing the LR every STEP_SIZE epochs
    scheduler = None
    if Config.SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=5, gamma=0.1
        )

    if Config.GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model_to_train.to(device)

    logger.info("Training commences")
    trained_model, *metrics = train_model(
        model=model_to_train,
        epochs=Config.EPOCHS,
        dataloaders=data_loaders,
        device=device,
        optimizer=optimizer,
        loss_function=loss_function,
        dataset_size=dataset_sizes,
        scheduler=scheduler,
    )
    path_to_weights = os.path.join(args.output_path, "model_weights.pth")
    torch.save(trained_model, path_to_weights)  # This is bad
    logger.info("Model saved")

    (
        val_accuracy_history,
        val_loss_history,
        best_val_accuracy,
        best_accuracy_epoch,
    ) = metrics
    visualise_training_results(val_accuracy_history, val_loss_history)
    logger.info(
        f"Best accuracy: {best_val_accuracy} achieved "
        f"on the {best_accuracy_epoch} epoch"
    )
    return 0


if __name__ == "__main__":
    main()
