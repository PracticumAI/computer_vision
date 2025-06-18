import requests
import os
import time
import tarfile

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt

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

    while not os.path.exists(filename):
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

    # Define the image size using the 1st 2 elements of the shape parameter
    # We don't need the number of channels here, just the dimensions to use
    image_size = shape[:2]

    # Get the class names
    class_names = os.listdir(path)

    images = []  # Initialize the images list
    labels = []  # Initialize the labels list
    cls_counts = {}

    # Get the images and labels to use for training and validation
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            images.append(image_path)
            labels.append(class_name)

    # Print the number of number of images per class
    print("\nFor the full dataset: ")
    print("   Class          # of images     # of total")
    print("--------------------------------------------")
    for class_name in class_names:
        print(
            f"{class_name:>15} {labels.count(class_name):11}"
            f"         {labels.count(class_name)/len(labels)*100:.1f}%"
        )
        # Save class count to return if requested
        cls_counts[class_name] = labels.count(class_name)
    print("--------------------------------------------")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if stratify: 
        print("Splitting the dataset with stratification, this may take a bit.")

        # Split the data using stratification
        image_train, image_val, labels_train, labels_val  = train_test_split(
            images, labels, test_size=0.2, stratify=labels, random_state=123
        )

        # Load the dataset using ImageFolder
        dataset = ImageFolder(path, transform=transform)

        # Create the dataloader
        data_train = DataLoader(image_train, labels_train, transform=transform)
        data_val = DataLoader(image_val, labels_val, transform=transform)

        # Get the class counts for the training data
        train_class_counts = []
        for class_name in class_names:
            train_class_counts.append(data_train.count(class_name))

        # Print the number of number of images per class
        print("\nFor the stratified split training dataset: ")
        print("   Class          # of images     # of total")
        print("--------------------------------------------")
 
        for class_name, count in zip(data_train.class_names, train_class_counts):
            print(
                 f"{class_name:>15} {count.numpy():11.0f}"
                 f"         {count.numpy()/len(train_labels)*100:.1f}%"
             )
            
    else:
        # Split the data randomly
        # Load the dataset using ImageFolder
        dataset = ImageFolder(path, transform=transform)

        # Calculate the sizes of training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders for training and validation sets
        data_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
        data_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
        

    if show_pictures:
        # Get the class names
        class_names = list(data_train.class_names)
        print(f'The classes in your dataset are: {class_names}')

        # Display up to 3 images from each of the categories
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 10))

            # Get one batch to use for display
            for images, labels in data_train.take(1):
                break

            # Find indices of the desired class in this batch
            mask = labels == i
            # Select images of the desired class
            class_images = images[mask]

            # Number of images to show.
            # We don't want to show more than 3 images.
            num_images = min(len(class_images), 3)

            for j in range(num_images):
                ax = plt.subplot(1, num_images, j + 1)
                plt.imshow(class_images[j].astype("uint8"))
                plt.title(class_name)
                plt.axis("off")
            plt.show()

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return data_train, data_val, cls_counts

    return data_train, data_val, labels_train, labels_val 


def make_model(input_shape=(3, 80, 80), num_classes=4):
    """Create a simple CNN model using PyTorch"""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 20 * 20, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 20 * 20)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return SimpleCNN()

def compile_train_model(train_loader, val_loader, model, loss_fn, optimizer, num_epochs=10, device='cuda'):
    """Compile and train the model"""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader)}")

    return model

def evaluate_model(test_loader, model, device='cuda'):
    """Evaluate the model"""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")