"""Module to help with the data loading, model training and evaluation for the
   Bee vs Wasp exercise"""

import requests
import os
import time
import datetime
import tarfile

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from keras.preprocessing.image import ImageDataGenerator
from keras.losses import SparseCategoricalCrossentropy

# Import the Sequential model: a linear stack of layers
# from Keras module in TensorFlow.
from keras.models import Sequential
# Import the Dense layer: a fully connected neural network layer
# from Keras module in TensorFlow.
from keras.layers import Dense
# Import the Flatten layer: used to convert input data into a 1D array
# from Keras module in TensorFlow.
from keras.layers import Flatten
from keras.utils import to_categorical
from keras import layers 
from keras import losses

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


def download_file(url, filename):
    """Download a file from a URL and save it to the current directory"""

    # Download the file using requests
    try:
        response = requests.get(url, stream=True, timeout=60)
    except requests.exceptions.RequestException as e:
        # Print an error message if the download fails
        print(f"Failed to download {url}: {e}")
        return

    # Create a file object and write the response content in chunks
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Wait for the file to finish downloading
    while not os.path.exists(filename):
        time.sleep(1)

    # Print a success message
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
        os.path.normpath(f"/blue/practicum-ai/share/data/{folder_name}"),
        os.path.normpath(f"/project/scinet_workshop2/data/{folder_name}"),
        os.path.join("data", folder_name),
        os.path.normpath(folder_name),
    ]

    for path in likely_paths:
        if os.path.exists(path):
            print(f"Found data at {path}.")
            return path

    prompt = (
        "Could not find data in the common locations. "
        "Do you know the path? (yes/no): "
    )
    answer = input(prompt)

    if answer.lower() == "yes":
        user_input = input("Please enter the path to the data folder: ")
        path = os.path.join(os.path.normpath(user_input), folder_name)
        if os.path.exists(path):
            print(f"Thanks! Found your data at {path}.")
            return path

        print("Sorry, that path does not exist.")

    answer = input("Do you want to download the data? (yes/no): ")

    if answer.lower() == "yes":
        print("Downloading data, this may take a minute.")
        download_file(url, filename)
        print("Data downloaded, unpacking")
        extract_file(filename, dest)
        print(
            "Data downloaded and unpacked. Now available at "
            f"{os.path.join(dest, folder_name)}."
        )
        return os.path.normpath(os.path.join(dest, folder_name))

    print(
        "Sorry, I cannot find the data."
        "Please download it manually from"
        "https://www.dropbox.com/s/x70hm8mxqhe7fa6/bee_vs_wasp.tar.gz"
        "and unpack it to the data folder."
    )


def count_class(counts, batch, classes):
    """Count number of samples per class in a batch"""
    for i in range(classes):
        cc = tf.cast(batch[1] == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)

def filter_by_class(x, y, class_index):
    return tf.equal(y, class_index)

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
    
    if stratify: 
        print("Splitting the dataset with stratification, this may take a bit.")
        # Create an image dataset
        dataset = keras.utils.image_dataset_from_directory(
            path,
            batch_size=batch_size,
            image_size=image_size,
            labels='inferred',  # Assuming directory structure reflects classes
            label_mode='int',
           
            shuffle=False  # Do not shuffle to maintain order for splitting
        )

        images = []
        labels = []

        # Extract images and labels from the dataset
        for image_batch, label_batch in dataset:
            images.extend(image_batch.numpy())
            labels.extend(label_batch.numpy())

        images = np.array(images)
        labels = np.array(labels)
        
        # One hot encode the labels
        #labels_one_hot = to_categorical(labels, num_classes=4)

        # Use sklearn's train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Remake train and val datasets
        data_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        data_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Attach the class_names to the datasets.
        data_train.class_names = class_names
        data_val.class_names = class_names

        label_counts = {label: 0 for label in data_train.class_names}
        for images, labels in data_train:
            for label in labels.numpy():
                label_name = data_train.class_names[label]
                label_counts[label_name] += 1

        for label, count in label_counts.items():
            print(f"{label}: {count} images")

        # Print the number of number of images per class
        print("\nFor the stratified split training dataset: ")
        print("   Class          # of images     # of total")
        print("--------------------------------------------")
        for class_name in class_names:
            print(
                f"{class_name:>15} {label_counts[class_name]:11}"
                f"         {label_counts[class_name]/len(X_val)*100:.1f}%"
            )
        print("--------------------------------------------")

        
    else:
        # Split the data randomly
        data_train = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset='training',
        seed=123,
        labels='inferred',
        label_mode='int'
        )

        data_val = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            batch_size=batch_size,
            image_size=image_size,
            validation_split=0.2,
            subset='validation',
            seed=123,
            labels='inferred',
            label_mode='int'
        )

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
            class_images = tf.boolean_mask(images, mask)
            # Convert TensorFlow tensors to NumPy arrays
            class_images = np.array(class_images)

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

    return data_train, data_val


def load_optimizer(optimizer_name):
    """Takes an optimizer name as a string and checks if it's valid"""

    # Check if the optimizer name is valid
    if optimizer_name in tf.keras.optimizers.__dict__:
        # Return the corresponding optimizer function
        return tf.keras.optimizers.__dict__[optimizer_name]

    # Raise an exception if the optimizer name is invalid
    raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def make_model(
        activation='relu',
        shape=(80, 80, 3),
        num_classes=4,
        drouput_rate=0.2
        ):
    '''Sets up a model. 
          Takes in an activation function, shape for the input images, 
          and number of classes and dropout rate. Returns the model.'''
    print("*****************************************************************")
    print("Make model:")
    print(f"  - Using the activation function: {activation}.")
    print(f"  - Model will have {num_classes} classes.")
    print(f"  - Using a dropout rate of: {drouput_rate}.")
    print("*****************************************************************")

    # Define the model
    model = tf.keras.Sequential([
        layers.Input(shape=shape),
        layers.Rescaling(1./255),
        layers.Conv2D(
            32, (3, 3), padding='same', activation=activation
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drouput_rate),
        layers.Conv2D(128, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(drouput_rate),
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def compile_train_model(
    data_train,
    data_val,
    model,
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer="Adam",
    learning_rate=0.0001,
    epochs=10,
    weights=False,
    log_name=None,
    callbacks=None,
):
    """Compiles and trains the model.
    Takes in an data_train, data_val, model, loss function, optimizer,
    learning rate, epochs, if class weights should be used, and a list of
    callbacks. Returns the compiled model and training history."""

    # Deal with class weights
    num_classes = len(data_train.class_names)
    class_indices = range(num_classes)

    if not weights:
        # Create a list of 1s as long as number of classes
        cls_wt = [1] * num_classes 
        class_weights = dict(zip(class_indices, cls_wt))  

    else:
        # Calculate class weights to deal with imbalance
        class_names = list(data_train.class_indices.keys())
        print(class_names)
        # Make a y from cls_counts
        y_vals = []
        for cls in list(data_train.class_indices.keys()):
            y_vals += [data_train.class_indices[cls]] * int(weights[cls])

        cls_wt = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(y_vals), 
            y=y_vals
        )

        class_weights = dict(zip(class_indices, cls_wt))

    if callbacks is None:
        callbacks = []

    print("******************************************************************")
    print("Compile and Train the model:")
    print(f"  - Using the loss function: {loss}.")
    print(f"  - Using the optimizer: {optimizer}.")
    print(f"  - Using learning rate of: {learning_rate}.")
    print(f"  - Running for {epochs} epochs.")
    print(f"  - Using class weights: {class_weights}.")
    print(f"  - Using these callbacks: {callbacks}.")
    print("******************************************************************")

    # Compile the model
    opt = load_optimizer(optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    # Set name for the log directory
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit_{log_name}_{epochs}_{time}"
    callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(
        data_train,
        epochs=epochs,
        validation_data=data_val,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    return model, history


def evaluate_model(data_val, model, history, num_classes=4):
    """Evaluates a model.
    Takes in an data_train, data_val, model, history, number of classes."""

    print("******************************************************************")
    print("Evaluate the model:")
    print("******************************************************************")
    # Evaluate the model
    loss, accuracy = model.evaluate(data_val)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

    # Plot the training and validation loss over time
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy over time
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Get the class names
    class_names = list(data_val.class_names)
    
    # Create the Confusion Matrix
    print("Calculating the confusion matrix. This can take a bit.")
    # Step 1: Get predictions
    all_predictions = []
    all_labels = []

    for images, labels in data_val:
        # Get predictions for this batch
        predictions = model.predict(images, verbose=0)

        # Convert predictions to class indices
        pred_classes = np.argmax(predictions, axis=1)

        # Append to our lists
        all_predictions.extend(pred_classes)
        all_labels.extend(labels.numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Step 2: Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)
    plt.colorbar()
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()
