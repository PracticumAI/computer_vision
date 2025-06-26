import requests
import time
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

import os
import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import torchmetrics separately - it's now a separate package
try:
    from torchmetrics import Accuracy
except ImportError:
    # Create a simple accuracy metric if torchmetrics is not available
    class Accuracy:
        def __init__(self):
            pass

        def __call__(self, preds, target):
            return (preds == target).float().mean()


# For transfer learning with pre-trained models
try:
    import timm
except ImportError:
    print("Warning: timm not available. Transfer learning features will not work.")
    timm = None

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pathlib import Path
import pytorch_lightning as pl


class ImageDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        image = Image.open(image_path).convert("RGB")
        label = self.label_list[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def download_file(url, filename):
    """Download a file from a URL and save it to the current directory"""
    try:
        print(f"Requesting download from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        print(f"Writing to file: {filename}")
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Wait for file to be completely written
        max_wait = 10  # maximum 10 seconds
        wait_count = 0
        while not Path(filename).exists() and wait_count < max_wait:
            time.sleep(1)
            wait_count += 1

        if Path(filename).exists():
            print(f"Downloaded {filename} successfully.")
            return True
        else:
            print(f"Error: File {filename} was not created after download.")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False


def extract_file(filename, data_folder):
    import tarfile, os

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        return False
    try:
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(data_folder)
        print(f"Extracted {filename} to {data_folder}")
        # Print what was extracted
        print("Extracted folders:", os.listdir(data_folder))
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False


def manage_data(
    url="https://data.rc.ufl.edu/pub/practicum-ai/Computer_Vision/bee_vs_wasp_reduced.tar.gz",
    filename="bee_vs_wasp_reduced.tar.gz",
    folder_name="bee_vs_wasp_reduced",
    dest="data",
):
    """Try to find the data for the exercise and return the path"""

    # Check common paths of where the data might be on different systems
    likely_paths = [
        Path(f"/blue/practicum-ai/share/data/{folder_name}"),
        Path(f"/project/scinet_workshop2/data/{folder_name}"),
        Path("data") / folder_name,
        Path(folder_name),
    ]

    for path in likely_paths:
        if path.exists() and path.is_dir():
            print(f"Found data at: {path}")
            return str(path)

    prompt = (
        "Could not find data in the common locations. "
        "Do you know the path? (yes/no): "
    )
    answer = input(prompt)

    if answer.lower() == "yes":
        path = input("Please enter the path to the data folder: ")
        if Path(path).exists():
            return path
        else:
            print("Path does not exist.")
            return None

    answer = input("Do you want to download the data? (yes/no): ")

    if answer.lower() == "yes":
        print(f"Downloading {filename}...")
        download_file(url, filename)
        extract_file(filename, dest)
        return str(Path(dest) / folder_name)

    print(
        "Sorry, I cannot find the data."
        "Please download it manually from"
        f"{url}"
        f"and unpack it to the {dest} folder."
    )

    return None


def manage_full_data(
    url="https://data.rc.ufl.edu/pub/practicum-ai/Computer_Vision/bee_vs_wasp.tar.gz",
    filename="bee_vs_wasp.tar.gz",
    folder_name="bee_vs_wasp",
    dest="data",
):
    """Try to find the data for the data imbalance exercise and return the path"""

    # Check common paths of where the data might be on different systems
    likely_paths = [
        Path(f"/blue/practicum-ai/share/data/{folder_name}"),
        Path(f"/project/scinet_workshop2/data/{folder_name}"),
        Path("data") / folder_name,
        Path(folder_name),
    ]

    for path in likely_paths:
        if path.exists() and path.is_dir():
            print(f"Found data at: {path}")
            return str(path)

    prompt = (
        "Could not find data in the common locations. "
        "Do you know the path? (yes/no): "
    )
    answer = input(prompt)

    if answer.lower() == "yes":
        path = input("Please enter the path to the data folder: ")
        if Path(path).exists():
            return path
        else:
            print("Path does not exist.")
            return None

    answer = input("Do you want to download the data? (yes/no): ")

    if answer.lower() == "yes":
        print(f"Downloading {filename}...")
        download_file(url, filename)
        extract_file(filename, dest)
        return str(Path(dest) / folder_name)

    print(
        "Sorry, I cannot find the data."
        "Please download it manually from"
        f"{url}"
        f"and unpack it to the {dest} folder."
    )

    return None


class BeeWaspDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Bee vs Wasp dataset"""

    def __init__(
        self,
        data_path,
        batch_size=32,
        shape=(80, 80, 3),
        train_split=0.8,
        num_workers=4,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape
        self.train_split = train_split
        self.num_workers = num_workers

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(shape[:2]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        if self.dataset is None:
            self.dataset = ImageFolder(self.data_path, transform=self.transform)

        if stage == "fit" or stage is None:
            # Split dataset
            train_size = int(self.train_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),  # For reproducibility
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        # Use validation set as test set for now
        return self.val_dataloader()

    def get_class_info(self):
        """Get class names and counts"""
        if self.dataset is None:
            self.dataset = ImageFolder(self.data_path, transform=self.transform)

        class_names = self.dataset.classes
        cls_counts = {class_name: 0 for class_name in class_names}

        # Count images per class
        for _, label_idx in self.dataset.samples:
            class_name = class_names[label_idx]
            cls_counts[class_name] += 1

        return class_names, cls_counts


def load_display_data(
    path,
    batch_size=32,
    shape=(80, 80, 3),
    show_pictures=True,
    return_cls_counts=False,
    train_split=0.8,
    num_workers=4,
):
    """Creates a PyTorch Lightning DataModule and optionally displays sample images"""
    print("******************************************************************")
    print("Load data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Train/validation split: {train_split:.1%}/{1-train_split:.1%}")
    print(f"  - Using {num_workers} workers for data loading")
    print(f"  - Returning class counts for later use? {return_cls_counts}")
    print("******************************************************************")

    # Create DataModule
    data_module = BeeWaspDataModule(
        data_path=path,
        batch_size=batch_size,
        shape=shape,
        train_split=train_split,
        num_workers=num_workers,
    )

    # Setup the data module
    data_module.setup("fit")

    # Get class information
    class_names, cls_counts = data_module.get_class_info()

    # Print class distribution
    total_images = len(data_module.dataset)
    print("\nFor the full dataset: ")
    print("   Class          # of images     # of total")
    print("--------------------------------------------")
    for class_name in class_names:
        count = cls_counts[class_name]
        percentage = count / total_images * 100
        print(f"{class_name:>15} {count:11}         {percentage:.1f}%")
    print("--------------------------------------------")

    if show_pictures:
        print(f"The classes in your dataset are: {class_names}")

        # Get a batch from the training dataloader for display
        train_loader = data_module.train_dataloader()
        images, labels = next(iter(train_loader))

        # Convert to numpy and denormalize for display
        images = images.numpy()

        # Create subplots: one row per class, 3 columns per row
        fig, axes = plt.subplots(
            len(class_names), 3, figsize=(12, 3 * len(class_names))
        )

        # Handle case where there's only one class (axes would be 1D)
        if len(class_names) == 1:
            axes = axes.reshape(1, -1)

        # Track how many images we've shown per class
        shown_per_class = {class_name: 0 for class_name in class_names}

        # Go through images and place them in the grid
        for img, label in zip(images, labels):
            class_name = class_names[label.item()]
            class_idx = label.item()

            # Only show up to 3 images per class
            if shown_per_class[class_name] < 3:
                col_idx = shown_per_class[class_name]

                # Denormalize image for display
                img_display = img.transpose(1, 2, 0)
                img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array(
                    [0.485, 0.456, 0.406]
                )
                img_display = np.clip(img_display, 0, 1)

                # Display the image
                axes[class_idx, col_idx].imshow(img_display)
                axes[class_idx, col_idx].set_title(f"{class_name}")
                axes[class_idx, col_idx].axis("off")

                shown_per_class[class_name] += 1

        # Hide any unused subplots (if we don't have enough images for some classes)
        for class_idx in range(len(class_names)):
            for col_idx in range(3):
                class_name = class_names[class_idx]
                if shown_per_class[class_name] <= col_idx:
                    axes[class_idx, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return data_module, cls_counts

    return data_module


class SimpleCNN(pl.LightningModule):
    """Simple CNN model using PyTorch Lightning"""

    def __init__(
        self,
        num_classes=4,
        learning_rate=0.001,
        input_shape=(3, 80, 80),
        dropout_rate=0.0,
        conv_padding=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture with configurable padding and dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=conv_padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=conv_padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # No padding for pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # Dynamically calculate the correct input size for the first linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._get_conv_output(dummy_input)
            self.conv_output_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Metrics for tracking - updated API
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def _get_conv_output(self, x):
        """Helper method to calculate the output size after conv layers"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before final layer
        x = self.fc2(x)
        return x

    def training_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.train_accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.test_accuracy(outputs, labels)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Use the learning rate from hyperparameters
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Optional: Add learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def train_model(
    data_module,
    num_classes=4,
    learning_rate=0.001,
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    input_shape=(3, 80, 80),
    dropout_rate=0.5,
    conv_padding=1,
):
    """Train a CNN model using PyTorch Lightning

    Args:
        data_module: Lightning DataModule for the dataset
        num_classes: Number of classes for classification
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of training epochs
        accelerator: Device type ('auto', 'gpu', 'cpu')
        devices: Number/type of devices to use
        input_shape: Shape of input images (channels, height, width)
        dropout_rate: Dropout probability (0.0 to 1.0)
        conv_padding: Padding for convolutional layers

    Returns:
        tuple: (trained_model, trainer)
    """

    # Create model with configurable parameters
    model = SimpleCNN(
        num_classes=num_classes,
        learning_rate=learning_rate,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        conv_padding=conv_padding,
    )

    # Create logger for TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="cnn_experiment")

    # Create callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=False, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="best-checkpoint",
        save_top_k=1,
        mode="max",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    return model, trainer


def test_model(data_module, model, trainer=None):
    """Test the model using PyTorch Lightning and display evaluation plots"""

    if trainer is None:
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

    # Test the model
    results = trainer.test(model, datamodule=data_module)

    # Plot training and validation metrics if available
    try:
        # Extract metrics from the model's logged history
        if hasattr(model, "trainer") and model.trainer is not None:
            # Check if we can access the logger's metrics
            if hasattr(model.trainer, "logger") and model.trainer.logger is not None:
                logger = model.trainer.logger

                # For TensorBoard logger, we can access the log directory
                if hasattr(logger, "log_dir"):
                    try:
                        from tensorboard.backend.event_processing.event_accumulator import (
                            EventAccumulator,
                        )

                        # Create event accumulator to read the tensorboard logs
                        event_acc = EventAccumulator(logger.log_dir)
                        event_acc.Reload()

                        # Get scalar tags
                        scalar_tags = event_acc.Tags()["scalars"]
                        print(f"Available scalar tags: {scalar_tags}")  # Debug print

                        # Initialize lists for metrics
                        train_losses = []
                        val_losses = []
                        train_accs = []
                        val_accs = []

                        # Extract training and validation metrics - focus on epoch-level only
                        for tag in scalar_tags:
                            tag_lower = tag.lower()
                            print(f"Processing tag: {tag}")  # Debug print

                            # Training loss - look for epoch-level metrics only (exclude step-level)
                            if (
                                "train" in tag_lower
                                and "loss" in tag_lower
                                and "step" not in tag_lower
                            ):
                                events = event_acc.Scalars(tag)
                                train_losses = [(e.step, e.value) for e in events]
                                print(
                                    f"Found train loss: {tag} with {len(train_losses)} points"
                                )

                            # Validation loss - these are typically epoch-level by default
                            elif (
                                "val" in tag_lower
                                and "loss" in tag_lower
                                and "step" not in tag_lower
                            ):
                                events = event_acc.Scalars(tag)
                                val_losses = [(e.step, e.value) for e in events]
                                print(
                                    f"Found val loss: {tag} with {len(val_losses)} points"
                                )

                            # Training accuracy - look for epoch-level metrics only (exclude step-level)
                            elif (
                                "train" in tag_lower
                                and "acc" in tag_lower
                                and "step" not in tag_lower
                            ):
                                events = event_acc.Scalars(tag)
                                train_accs = [(e.step, e.value) for e in events]
                                print(
                                    f"Found train acc: {tag} with {len(train_accs)} points"
                                )

                            # Validation accuracy - these are typically epoch-level by default
                            elif (
                                "val" in tag_lower
                                and "acc" in tag_lower
                                and "step" not in tag_lower
                            ):
                                events = event_acc.Scalars(tag)
                                val_accs = [(e.step, e.value) for e in events]
                                print(
                                    f"Found val acc: {tag} with {len(val_accs)} points"
                                )

                    except ImportError:
                        print(
                            "TensorBoard not available for reading logs. Skipping training plots."
                        )
                        train_losses = val_losses = train_accs = val_accs = []
                    except Exception as e:
                        print(f"Could not read TensorBoard logs: {e}")
                        train_losses = val_losses = train_accs = val_accs = []

        # Plot if we have data
        if train_losses or val_losses or train_accs or val_accs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot loss
            loss_plotted = False
            if train_losses:
                epochs, losses = zip(*train_losses)
                ax1.plot(epochs, losses, label="Training Loss", marker="o", linewidth=2)
                loss_plotted = True
            if val_losses:
                epochs, losses = zip(*val_losses)
                ax1.plot(
                    epochs, losses, label="Validation Loss", marker="s", linewidth=2
                )
                loss_plotted = True

            if loss_plotted:
                ax1.set_title("Training and Validation Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No loss data available",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("Training and Validation Loss")

            # Plot accuracy
            acc_plotted = False
            if train_accs:
                epochs, accs = zip(*train_accs)
                ax2.plot(
                    epochs, accs, label="Training Accuracy", marker="o", linewidth=2
                )
                acc_plotted = True
            if val_accs:
                epochs, accs = zip(*val_accs)
                ax2.plot(
                    epochs, accs, label="Validation Accuracy", marker="s", linewidth=2
                )
                acc_plotted = True

            if acc_plotted:
                ax2.set_title("Training and Validation Accuracy")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No accuracy data available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Training and Validation Accuracy")

            plt.tight_layout()
            plt.show()
        else:
            print("Training metrics not available for plotting.")
            print("This can happen if the model wasn't trained in this session.")

    except Exception as e:
        print(f"Could not plot training metrics: {e}")
        print("This is normal if training metrics aren't available.")

    # Generate confusion matrix for validation data
    try:
        model.eval()
        all_predictions = []
        all_labels = []

        # Get predictions from validation data
        val_loader = data_module.val_dataloader()
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Get class names
        class_names, _ = data_module.get_class_info()

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Validation Data")
        plt.colorbar()

        # Add labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

        # Calculate and print per-class metrics
        print("\nPer-class Performance:")
        print("-" * 50)
        for i, class_name in enumerate(class_names):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            print(
                f"{class_name:>15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
            )

    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    return results


class BeeWaspAugmentedDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Bee vs Wasp dataset with data augmentation"""

    def __init__(
        self,
        data_path,
        batch_size=32,
        shape=(80, 80, 3),
        train_split=0.8,
        num_workers=4,
        augmentation_strength="light",
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape
        self.train_split = train_split
        self.num_workers = num_workers
        self.augmentation_strength = augmentation_strength

        # Define different augmentation levels
        if augmentation_strength == "none":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(shape[:2]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif augmentation_strength == "light":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((shape[0], shape[1])),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif augmentation_strength == "medium":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((shape[0], shape[1])),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomResizedCrop(
                        size=(shape[0], shape[1]), scale=(0.8, 1.0)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif augmentation_strength == "heavy":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((shape[0], shape[1])),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    ),
                    transforms.RandomResizedCrop(
                        size=(shape[0], shape[1]), scale=(0.7, 1.0)
                    ),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Validation transform (no augmentation)
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((shape[0], shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        if stage == "fit" or stage is None:
            # Create separate datasets with different transforms
            full_dataset = ImageFolder(self.data_path)

            # Split indices
            train_size = int(self.train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_indices, val_indices = random_split(
                range(len(full_dataset)),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            # Create training dataset with augmentation
            train_samples = [full_dataset.samples[i] for i in train_indices]
            self.train_dataset = ImageFolder(
                self.data_path, transform=self.train_transform
            )
            self.train_dataset.samples = train_samples
            self.train_dataset.targets = [s[1] for s in train_samples]

            # Create validation dataset without augmentation
            val_samples = [full_dataset.samples[i] for i in val_indices]
            self.val_dataset = ImageFolder(self.data_path, transform=self.val_transform)
            self.val_dataset.samples = val_samples
            self.val_dataset.targets = [s[1] for s in val_samples]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        # Use validation set as test set for now
        return self.val_dataloader()

    def get_class_info(self):
        """Get class names and counts"""
        full_dataset = ImageFolder(self.data_path)
        class_names = full_dataset.classes
        cls_counts = {class_name: 0 for class_name in class_names}

        # Count images per class
        for _, label_idx in full_dataset.samples:
            class_name = class_names[label_idx]
            cls_counts[class_name] += 1

        return class_names, cls_counts


def load_display_data_augmented(
    path,
    batch_size=32,
    shape=(80, 80, 3),
    show_pictures=True,
    return_cls_counts=False,
    train_split=0.8,
    num_workers=4,
    augmentation_strength="light",
):
    """Creates a PyTorch Lightning DataModule with data augmentation and optionally displays sample images"""
    print("******************************************************************")
    print("Load data with augmentation:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Train/validation split: {train_split:.1%}/{1-train_split:.1%}")
    print(f"  - Using {num_workers} workers for data loading")
    print(f"  - Augmentation strength: {augmentation_strength}")
    print(f"  - Returning class counts for later use? {return_cls_counts}")
    print("******************************************************************")

    # Create DataModule with augmentation
    data_module = BeeWaspAugmentedDataModule(
        data_path=path,
        batch_size=batch_size,
        shape=shape,
        train_split=train_split,
        num_workers=num_workers,
        augmentation_strength=augmentation_strength,
    )

    # Setup the data module
    data_module.setup("fit")

    # Get class information
    class_names, cls_counts = data_module.get_class_info()

    # Print class distribution
    total_images = sum(cls_counts.values())
    print("\nFor the full dataset: ")
    print("   Class          # of images     # of total")
    print("--------------------------------------------")
    for class_name in class_names:
        count = cls_counts[class_name]
        percentage = count / total_images * 100
        print(f"{class_name:>15} {count:11}         {percentage:.1f}%")
    print("--------------------------------------------")

    if show_pictures:
        print(f"The classes in your dataset are: {class_names}")

        # Show comparison between original and augmented images
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))

        # Get a batch from training data (with augmentation)
        train_loader = data_module.train_dataloader()
        aug_images, aug_labels = next(iter(train_loader))

        # Get a batch from validation data (without augmentation)
        val_loader = data_module.val_dataloader()
        orig_images, orig_labels = next(iter(val_loader))

        # Show first 6 images
        for i in range(6):
            # Original images (top row)
            img = orig_images[i].numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Original: {class_names[orig_labels[i]]}")
            axes[0, i].axis("off")

            # Augmented images (bottom row)
            img = aug_images[i].numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = np.clip(img, 0, 1)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Augmented: {class_names[aug_labels[i]]}")
            axes[1, i].axis("off")

        plt.suptitle(
            f"Data Augmentation Comparison (Strength: {augmentation_strength})"
        )
        plt.tight_layout()
        plt.show()

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return data_module, cls_counts

    return data_module


class TransferLearningCNN(pl.LightningModule):
    """Transfer Learning model using pre-trained networks with PyTorch Lightning"""

    def __init__(
        self,
        num_classes=4,
        learning_rate=0.0001,
        model_name="efficientnet_b5",
        freeze_backbone=False,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove the classifier head
            global_pool="",  # Remove global pooling
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone ({model_name}) frozen for feature extraction")
        else:
            print(f"Backbone ({model_name}) unfrozen for fine-tuning")

        # Get feature dimension from backbone
        with torch.no_grad():
            # Create a dummy input to get feature dimensions
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            if (
                len(dummy_features.shape) == 4
            ):  # If output is 4D (batch, channels, height, width)
                feature_dim = dummy_features.shape[1]
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
            else:  # If output is already 2D (batch, features)
                feature_dim = dummy_features.shape[1]
                self.global_pool = nn.Identity()
                self.flatten = nn.Identity()

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Metrics for tracking - updated API
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        print(
            f"Model created with {sum(p.numel() for p in self.parameters()):,} total parameters"
        )
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)

        # Apply global pooling if needed
        features = self.global_pool(features)
        features = self.flatten(features)

        # Apply classifier
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.train_accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Update and log metrics
        self.test_accuracy(outputs, labels)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Use different learning rates for backbone and classifier if fine-tuning
        if self.hparams.freeze_backbone:
            # Only optimize classifier parameters
            optimizer = optim.Adam(
                self.classifier.parameters(), lr=self.hparams.learning_rate
            )
        else:
            # Use different learning rates for backbone and classifier
            backbone_lr = (
                self.hparams.learning_rate * 0.1
            )  # Lower LR for pre-trained layers
            classifier_lr = self.hparams.learning_rate

            optimizer = optim.Adam(
                [
                    {"params": self.backbone.parameters(), "lr": backbone_lr},
                    {"params": self.classifier.parameters(), "lr": classifier_lr},
                ]
            )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def train_transfer_model(
    data_module,
    num_classes=4,
    learning_rate=0.0001,
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    model_name="efficientnet_b5",
    freeze_backbone=False,
    dropout_rate=0.3,
):
    """Train a transfer learning model using PyTorch Lightning

    Args:
        data_module: Lightning DataModule for the dataset
        num_classes: Number of classes for classification
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of training epochs
        accelerator: Device type ('auto', 'gpu', 'cpu')
        devices: Number/type of devices to use
        model_name: Name of the pre-trained model to use
        freeze_backbone: Whether to freeze the backbone for feature extraction
        dropout_rate: Dropout probability (0.0 to 1.0)

    Returns:
        tuple: (trained_model, trainer)
    """

    # Create transfer learning model
    model = TransferLearningCNN(
        num_classes=num_classes,
        learning_rate=learning_rate,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )

    # Create logger for TensorBoard
    logger = TensorBoardLogger("lightning_logs", name=f"transfer_{model_name}")

    # Create callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,  # More patience for transfer learning
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename=f"best-{model_name}-checkpoint",
        save_top_k=1,
        mode="max",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    return model, trainer


class BeeWaspWeightedDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Bee vs Wasp dataset with weighted sampling for class imbalance"""

    def __init__(
        self,
        data_path,
        batch_size=32,
        shape=(80, 80, 3),
        train_split=0.8,
        num_workers=4,
        use_weighted_sampler=True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape
        self.train_split = train_split
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((shape[0], shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        # Load full dataset
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

        # Get class names and counts
        self.class_names = self.dataset.classes
        self.class_counts = {cls: 0 for cls in self.class_names}

        # Count samples per class
        for _, label in self.dataset:
            class_name = self.class_names[label]
            self.class_counts[class_name] += 1

        # Split dataset
        train_size = int(self.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Calculate class weights for weighted sampling
        if self.use_weighted_sampler:
            # Get labels for training set
            train_labels = [self.dataset[idx][1] for idx in self.train_dataset.indices]

            # Calculate class weights (inverse frequency)
            class_counts = torch.bincount(torch.tensor(train_labels))
            total_samples = len(train_labels)
            class_weights = total_samples / (len(class_counts) * class_counts.float())

            # Create sample weights
            self.sample_weights = [class_weights[label] for label in train_labels]

    def train_dataloader(self):
        if self.use_weighted_sampler:
            # Use WeightedRandomSampler for balanced sampling
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def get_class_info(self):
        return self.class_names, self.class_counts


def load_imbalanced_data(
    path,
    batch_size=32,
    shape=(80, 80, 3),
    show_pictures=True,
    train_split=0.8,
    num_workers=4,
    use_weighted_sampler=True,
    show_class_distribution=True,
):
    """Creates a PyTorch Lightning DataModule for handling imbalanced data

    Args:
        path: Path to the dataset
        batch_size: Batch size for data loaders
        shape: Target shape for images (height, width, channels)
        show_pictures: Whether to display sample images
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        use_weighted_sampler: Whether to use weighted sampling for class balance
        show_class_distribution: Whether to print the class distribution

    Returns:
        BeeWaspWeightedDataModule: Configured data module

    Raises:
        FileNotFoundError: If the dataset path doesn't exist
        ValueError: If the path is None or invalid
    """
    # Validate the path
    if path is None:
        raise ValueError(
            "Dataset path cannot be None. Please provide a valid path to the dataset."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    print("******************************************************************")
    print("Load imbalanced data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Train/validation split: {train_split:.1%}/{1-train_split:.1%}")
    print(f"  - Using {num_workers} workers for data loading")
    print(f"  - Using weighted sampler: {use_weighted_sampler}")
    print("******************************************************************")

    # Create DataModule
    data_module = BeeWaspWeightedDataModule(
        data_path=path,
        batch_size=batch_size,
        shape=shape,
        train_split=train_split,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )

    # Setup the data module
    data_module.setup("fit")

    # Get class information
    class_names, class_counts = data_module.get_class_info()

    # Print class distribution only if requested
    if show_class_distribution:
        total_images = sum(class_counts.values())
        print("\nClass distribution in the full dataset:")
        print("   Class                # of images     % of total")
        print("--------------------------------------------------------")
        for class_name in class_names:
            count = class_counts[class_name]
            percentage = (count / total_images) * 100
            print(f"   {class_name:<15} {count:>8}     {percentage:>6.1f}%")
        print("--------------------------------------------------------")
        print(f"   Total               {total_images:>8}     100.0%")

    if show_pictures:
        # Show sample images from each class
        plt.figure(figsize=(15, 10))

        # Get one batch from training data
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        images, labels = batch

        # Show first few images from the batch
        num_show = min(12, len(images))
        for i in range(num_show):
            plt.subplot(3, 4, i + 1)

            # Denormalize image for display
            img = images[i].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)

            # Convert to numpy and transpose for matplotlib
            img_np = img.permute(1, 2, 0).numpy()

            plt.imshow(img_np)
            plt.title(f"Class: {class_names[labels[i]]}")
            plt.axis("off")

        if use_weighted_sampler:
            title = "Sample Images from Training Data (with Weighted Sampling)"
        else:
            title = "Sample Images from Training Data (Natural Distribution)"

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    return data_module
