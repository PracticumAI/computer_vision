import requests
import time
import tarfile

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def download_file(url, filename):
    """Download a file from a URL and save it to the current directory"""
    try:
        response = requests.get(url, stream=True, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    while not Path(filename).exists():
        time.sleep(1)

    print(f"Downloaded {filename} successfully.")


def extract_file(filename, data_folder):
    """Extract a tar file to a specified folder"""

    # Check if the file is a tar file
    if tarfile.is_tarfile(filename):
        # Open the tar file
        tar = tarfile.open(filename, "r:gz")
        # Extract all the files to the data folder, filter for security
        tar.extractall(data_folder, filter="data")
        # Close the tar file
        tar.close()
        # Print a success message
        print(f"Extracted {filename} to {data_folder} successfully.")
    else:
        # Print an error message
        print(f"{filename} is not a valid tar file.")


def manage_data(
    url="https://www.dropbox.com/s/x70hm8mxqhe7fa6/bee_vs_wasp.tar.gz?dl=1",
    filename="bee_vs_wasp.tar.gz",
    folder_name="bee_vs_wasp",
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
        if path.exists():
            print(f"Found data at {path}.")
            return str(path)

    prompt = (
        "Could not find data in the common locations. "
        "Do you know the path? (yes/no): "
    )
    answer = input(prompt)

    if answer.lower() == "yes":
        user_input = input("Please enter the path to the data folder: ")
        path = Path(user_input) / folder_name
        if path.exists():
            print(f"Thanks! Found your data at {path}.")
            return str(path)

        print("Sorry, that path does not exist.")

    answer = input("Do you want to download the data? (yes/no): ")

    if answer.lower() == "yes":
        print("Downloading data, this may take a minute.")
        download_file(url, filename)
        print("Data downloaded, unpacking")
        extract_file(filename, dest)
        final_path = Path(dest) / folder_name
        print(
            "Data downloaded and unpacked. Now available at "
            f"{final_path}."
        )
        return str(final_path)

    print(
        "Sorry, I cannot find the data."
        "Please download it manually from"
        "https://www.dropbox.com/s/x70hm8mxqhe7fa6/bee_vs_wasp.tar.gz"
        "and unpack it to the data folder."
    )

    return None


def load_display_data(
    path,
    batch_size=32,
    shape=(80, 80, 3),
    show_pictures=True,
    stratify=False,
    return_cls_counts=False,
):
    """Takes a path, batch size, target shape for images and optionally
    whether to show sample images. Returns training and validation datasets
    """
    print("******************************************************************")
    print("Load data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Stratify when sampling? {stratify}")
    print(f"  - Returning class counts for later use? {return_cls_counts}")
    print("******************************************************************")

    # Get the class names and count images
    path_obj = Path(path)
    class_names = [item.name for item in path_obj.iterdir() if item.is_dir()]
    
    images = []
    labels = []
    cls_counts = {}

    for class_name in class_names:
        class_path = path_obj / class_name
        class_images = [str(img) for img in class_path.iterdir() if img.is_file()]
        images.extend(class_images)
        labels.extend([class_name] * len(class_images))
        cls_counts[class_name] = len(class_images)

    # Print class distribution
    print("\nFor the full dataset: ")
    print("   Class          # of images     # of total")
    print("--------------------------------------------")
    for class_name in class_names:
        count = cls_counts[class_name]
        percentage = count / len(labels) * 100
        print(f"{class_name:>15} {count:11}         {percentage:.1f}%")
    print("--------------------------------------------")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(shape[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and create data loaders
    dataset = ImageFolder(path, transform=transform)
    
    # Use random split (stratification would require more complex implementation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    data_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if show_pictures:
        print(f'The classes in your dataset are: {dataset.classes}')

        # Display up to 3 images from each class
        for class_name in class_names:
            plt.figure(figsize=(10, 3))
            class_images = [img for img, label in zip(images, labels) if label == class_name]
            num_images = min(3, len(class_images))
            selected_images = np.random.choice(class_images, num_images, replace=False)

            for i, image_path in enumerate(selected_images):
                image = Image.open(image_path)
                plt.subplot(1, num_images, i + 1)
                plt.imshow(image)
                plt.axis("off")
                plt.title(class_name)
            plt.show()

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return data_train, data_val, cls_counts

    return data_train, data_val

class SimpleCNN(pl.LightningModule):
    """Simple CNN model using PyTorch Lightning"""
    
    def __init__(self, input_shape=(3, 80, 80), num_classes=4, learning_rate=0.001):
        super(SimpleCNN, self).__init__()
        self.save_hyperparameters()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 20 * 20)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_accuracy': accuracy, 'predictions': predicted, 'labels': labels}
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', accuracy, on_epoch=True)
        
        return {'test_loss': loss, 'test_accuracy': accuracy, 'predictions': predicted, 'labels': labels}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def make_model(input_shape=(3, 80, 80), num_classes=4, learning_rate=0.001):
    """Create a simple CNN model using PyTorch Lightning"""
    return SimpleCNN(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)

def compile_train_model(train_loader, val_loader, model=None, num_epochs=10, learning_rate=0.001, 
                       num_classes=4, accelerator='auto'):
    """Train the model using PyTorch Lightning"""
    
    if model is None:
        model = make_model(num_classes=num_classes, learning_rate=learning_rate)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    
    # Setup logger
    logger = TensorBoardLogger('lightning_logs/', name='bee_wasp_model')
    
    # Create trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=accelerator,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer

def evaluate_model(test_loader, model, trainer=None):
    """Evaluate the model using PyTorch Lightning"""
    
    if trainer is None:
        trainer = Trainer(accelerator='auto')
    
    # Test the model
    results = trainer.test(model, test_loader)
    
    # Print results
    if results:
        test_accuracy = results[0]['test_accuracy']
        test_loss = results[0]['test_loss']
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
    
    return results