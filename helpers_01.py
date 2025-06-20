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

class BeeWaspDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Bee vs Wasp dataset"""
    
    def __init__(self, data_path, batch_size=32, shape=(80, 80, 3), train_split=0.8, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape
        self.train_split = train_split
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        if self.dataset is None:
            self.dataset = ImageFolder(self.data_path, transform=self.transform)
        
        if stage == 'fit' or stage is None:
            # Split dataset
            train_size = int(self.train_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
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
        num_workers=num_workers
    )
    
    # Setup the data module
    data_module.setup('fit')
    
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
        print(f'The classes in your dataset are: {class_names}')

        # Get a batch from the training dataloader for display
        train_loader = data_module.train_dataloader()
        images, labels = next(iter(train_loader))
        
        # Convert to numpy and denormalize for display
        images = images.numpy()
        
        # Create subplots: one row per class, 3 columns per row
        fig, axes = plt.subplots(len(class_names), 3, figsize=(12, 3 * len(class_names)))
        
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
                img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_display = np.clip(img_display, 0, 1)
                
                # Display the image
                axes[class_idx, col_idx].imshow(img_display)
                axes[class_idx, col_idx].set_title(f"{class_name}")
                axes[class_idx, col_idx].axis('off')
                
                shown_per_class[class_name] += 1
        
        # Hide any unused subplots (if we don't have enough images for some classes)
        for class_idx in range(len(class_names)):
            for col_idx in range(3):
                class_name = class_names[class_idx]
                if shown_per_class[class_name] <= col_idx:
                    axes[class_idx, col_idx].axis('off')
        
        plt.tight_layout()
        plt.show()

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return data_module, cls_counts

    return data_module

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

def compile_train_model(data_module, model=None, num_epochs=10, learning_rate=0.001, 
                       num_classes=4, accelerator='auto'):
    """Train the model using PyTorch Lightning DataModule"""
    
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
    
    # Train the model using the DataModule
    trainer.fit(model, datamodule=data_module)
    
    return model, trainer

def evaluate_model(data_module, model, trainer=None):
    """Evaluate the model using PyTorch Lightning DataModule"""
    
    if trainer is None:
        trainer = Trainer(accelerator='auto')
    
    # Test the model using the DataModule
    results = trainer.test(model, datamodule=data_module)
    
    # Print results
    if results:
        test_accuracy = results[0]['test_accuracy']
        test_loss = results[0]['test_loss']
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
    
    return results