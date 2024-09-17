"""Module to help with the data loading, model training and evaluation for the
   Bee vs Wasp exercise"""
import os
import tarfile
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def download_file(url, filename):
    '''Download a file from a URL and save it to the current directory'''

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
    '''Extract a tar file to a specified folder'''

    # Check if the file is a tar file
    if tarfile.is_tarfile(filename):
        # Open the tar file
        tar = tarfile.open(filename, "r:gz")
        # Extract all the files to the data folder, filter for security
        tar.extractall(data_folder, filter='data')
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
    folder_name='bee_vs_wasp',
    dest='data'
):
    '''Try to find the data for the exercise and return the path'''

    # Check common paths of where the data might be on different systems
    likely_paths = [
        os.path.normpath(f'/blue/practicum-ai/share/data/{folder_name}'),
        os.path.normpath(f'/project/scinet_workshop2/data/{folder_name}'),
        os.path.join('data', folder_name),
        os.path.normpath(folder_name)
    ]

    for path in likely_paths:
        if os.path.exists(path):
            print(f'Found data at {path}.')
            return path

    prompt = (
        'Could not find data in the common locations. '
        'Do you know the path? (yes/no): '
    )
    answer = input(prompt)

    if answer.lower() == 'yes':
        user_input = input('Please enter the path to the data folder: ')
        path = os.path.join(os.path.normpath(user_input), folder_name)
        if os.path.exists(path):
            print(f'Thanks! Found your data at {path}.')
            return path

        print('Sorry, that path does not exist.')

    answer = input('Do you want to download the data? (yes/no): ')

    if answer.lower() == 'yes':
        print('Downloading data, this may take a minute.')
        download_file(url, filename)
        print('Data downloaded, unpacking')
        extract_file(filename, dest)
        print('Data downloaded and unpacked. Now available at '
              f'{os.path.join(dest, folder_name)}.')
        return os.path.normpath(os.path.join(dest, folder_name))

    print('Sorry, I cannot find the data.'
          'Please download it manually from'
          'https://www.dropbox.com/s/x70hm8mxqhe7fa6/bee_vs_wasp.tar.gz'
          'and unpack it to the data folder.')


def count_class(counts, batch, classes):
    '''Count number of samples per class in a batch'''
    for i in range(classes):
        cc = tf.cast(batch[1] == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)


def load_display_data(
    path,
    batch_size=32,
    shape=(80, 80, 3),
    show_pictures=True,
    stratify=False,
    return_cls_counts=False
):
    '''Takes a path, batch size, target shape for images and optionally
       whether to show sample images. Returns training and testing datasets
    '''
    print("******************************************************************")
    print("Load data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Returning class counts for later use? {return_cls_counts}")
    print("******************************************************************")
    # Define the directory path
    directory_path = path

    # Define the image size using the 1st 2 elements of the shape parameter
    # We don't need the number of channels here, just the dimensions to use
    image_size = shape[:2]

    # Load the dataset
    data_train = tf.keras.preprocessing.image_dataset_from_directory(
        directory_path,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset='training',
        seed=123,
        labels='inferred',
        label_mode='int'
    )

    data_test = tf.keras.preprocessing.image_dataset_from_directory(
        directory_path,
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
        class_names = data_train.class_names
        print(class_names)

        # Display up to 3 images from each of the categories
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 10))
            for images, labels in data_train.take(2):
                images = images.numpy()
                labels = labels.numpy()

                # Filter images of the current class
                class_images = images[labels == i]

                # Number of images to show. Limited by number
                #  of this class in the batch or specific number
                num_images = min(len(class_images), 3)

                for j in range(num_images):
                    ax = plt.subplot(1, num_images, j + 1)
                    plt.imshow(class_images[j].astype("uint8"))
                    plt.title(class_name)
                    plt.axis("off")
            plt.show()

    # Get and print counts by class
    print("\nGetting number of images per class. This may take a bit...")
    # Initialize counts
    counts = [0] * len(data_train.class_names)
    class_names = list(data_train.class_names)

    # Iterate through the training dataset batch by batch
    for batch in data_train:
        count_class(counts, batch, len(data_train.class_names))

    total = sum(counts)

    cls_counts = {}
    # Print the counts
    print("\nFor the Training set:")
    for i, count in enumerate(counts):
        print(f"Category {class_names[i]}: {count} images or "
              f"{count/total*100:.1f}% of total images.")
        cls_counts[i] = count

    if return_cls_counts:
        return data_train, data_test, cls_counts
    return data_train, data_test


def load_optimizer(optimizer_name):
    '''Takes an optimizer name as a string and checks if it's valid'''

    # Check if the optimizer name is valid
    if optimizer_name in tf.keras.optimizers.__dict__:
        # Return the corresponding optimizer function
        return tf.keras.optimizers.__dict__[optimizer_name]

    # Raise an exception if the optimizer name is invalid
    raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def compile_train_model(data_train, data_test, model,
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        optimizer='Adam', learning_rate=0.0001, epochs=10,
                        weights=False, callbacks=None):
    '''Compiles and trains the model.
        Takes in an data_train, data_test, model, loss function, optimizer,
        learning rate, epochs, if class weights should be used, and a list of
        callbacks. Returns the compiled model and training history.'''

    if callbacks is None:
        callbacks = []

    num_classes = len(data_train.class_names)

    if weights:
        # Calculate Class Weights to manage imbalance in the dataset
        print("Feature not yet implemented....sorry!")
    else:
        # Create a list of 1s as long as number of classes
        weight_list = [1] * num_classes
        class_indices = range(num_classes)
        class_weight = dict(zip(class_indices, weight_list))

    print("******************************************************************")
    print("Compile and Train the model:")
    print(f"  - Using the loss function: {loss}.")
    print(f"  - Using the optimizer: {optimizer}.")
    print(f"  - Using learning rate of: {learning_rate}.")
    print(f"  - Running for {epochs} epochs.")
    print(f"   -Using class weights: {class_weight})")
    print(f"  - Using these callbacks: {callbacks}")
    print("******************************************************************")
    # Compile the model

    opt = load_optimizer(optimizer)(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        data_train,
        epochs=epochs,
        validation_data=data_test,
        class_weight=class_weight,
        callbacks=[callbacks]
    )

    return model, history


def evaluate_model(data_train, data_test, model, history, num_classes=4):
    '''Evaluates a model.
    Takes in an data_train, data_test, model, history, number of classes.'''

    print("******************************************************************")
    print("Evaluate the model:")
    print("******************************************************************")
    # Evaluate the model
    loss, accuracy = model.evaluate(data_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

    # Plot the training and validation loss over time
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy over time
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Get the class names
    class_names = data_test.class_names

    # Make predictions on the test set
    y_pred = np.argmax(model.predict(data_test), axis=-1)

    # Get the true labels
    y_true = np.concatenate([y for x, y in data_test], axis=0)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)
    plt.colorbar()
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()
