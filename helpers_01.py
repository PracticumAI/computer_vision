import requests
import os
import time
import datetime
import tarfile

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class CustomDataset:
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names

    def __getattr__(self, name):
        return getattr(self.dataset, name)

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