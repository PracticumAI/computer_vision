import requests
import os
import time
import tarfile
import zipfile
import numpy as np
import matplotlib.pyplot as plt

def download_file(url="https://www.dropbox.com/scl/fi/ioupfqya76b7p8m1v1kdc/fruits_detection.zip?rlkey=ofgre83fdxa98p7ity8j9z8ip&st=atv7sz18&dl=1", filename="fruits_detection.zip"):

    # Download the file using requests
    response = requests.get(url, stream=True)

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
    if zipfile.is_zipfile(filename):
        # Open the zip file
        with zipfile.ZipFile(filename, "r") as zip_ref:
            # Extract all the files to the data folder
            zip_ref.extractall(data_folder)
            # Print a success message
            print(f"Extracted {filename} to {data_folder} successfully.")
    else:
        # Print an error message
        print(f"{filename} is not a valid tar or zip file.")
    
def manage_data(url="https://www.dropbox.com/scl/fi/ioupfqya76b7p8m1v1kdc/fruits_detection.zip?rlkey=ofgre83fdxa98p7ity8j9z8ip&st=atv7sz18&dl=1", filename="fruits_detection.zip", folder_name='fruits_detection', dest='data'):
    '''Try to find the data for the exercise and return the path'''
    
    # Check common paths of where the data might be on different systems
    likely_paths= [os.path.normpath(f'/blue/practicum-ai/share/data/{folder_name}'),
                   os.path.normpath(f'/project/scinet_workshop2/data/{folder_name}'),
                   os.path.join('data', folder_name),
                   os.path.normpath(folder_name)]
    
    for path in likely_paths:
        if os.path.exists(path):
            print(f'Found data at {path}.')
            return path

    answer = input(f'Could not find data in the common locations. Do you know the path? (yes/no): ')

    if answer.lower() == 'yes':
        path = os.path.join(os.path.normpath(input('Please enter the path to the data folder: ')),folder_name)
        if os.path.exists(path):
            print(f'Thanks! Found your data at {path}.')
            return path
        else:
            print(f'Sorry, that path does not exist.')
    
    answer = input('Do you want to download the data? (yes/no): ')

    if answer.lower() == 'yes':
        print('Downloading data, this may take a minute.')
        download_file(url, filename)
        print('Data downloaded, unpacking')
        extract_file(filename, dest)
        print(f'Data downloaded and unpacked. Now available at {os.path.join(dest,folder_name)}.')
        return os.path.normpath(os.path.join(dest,folder_name))   

    print('Sorry, I cannot find the data. Please download it manually from https://www.dropbox.com/scl/fi/ioupfqya76b7p8m1v1kdc/fruits_detection.zip and unpack it to the data folder.')      


def load_display_data(path, batch_size=32, shape=(80,80,3), show_pictures=True):
    '''Takes a path, batch size, target shape for images and optionally whether to show sample images.
       Returns training and testing datasets
    '''
    print("***********************************************************************")
    print("Load data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print("***********************************************************************")
    # Define the directory path
    directory_path = path
    
    # Define the batch size
    batch_size = batch_size
    
    # Define the image size using the 1st 2 elements of the shape parameter
    # We don't need the number of channels here, just the dimensions to use
    image_size = shape[:2]
    
    # Load the dataset
    X_train = tf.keras.preprocessing.image_dataset_from_directory(
        directory_path,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset='training',
        seed=123,
        labels='inferred',
        label_mode='int'
    )
    
    X_test = tf.keras.preprocessing.image_dataset_from_directory(
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
        class_names = X_train.class_names
        print(class_names)

        # Display up to 3 images from each of the categories
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 10))
            for images, labels in X_train.take(2):
                images = images.numpy()
                labels = labels.numpy()

                # Filter images of the current class
                class_images = images[labels == i]
                
                # Number of images to show.
                # Limited by number of this class in the batch or specific number
                num_images = min(len(class_images), 3)
                
                for j in range(num_images):
                    ax = plt.subplot(1, num_images, j + 1)
                    plt.imshow(class_images[j].astype("uint8"))
                    plt.title(class_name)
                    plt.axis("off")
            plt.show()
    return X_train, X_test

def load_optimizer(optimizer_name):
    '''Takes an optimizer name as a string and checks if it's valid'''

    # Check if the optimizer name is valid
    if optimizer_name in tf.keras.optimizers.__dict__:
        # Return the corresponding optimizer function
        return tf.keras.optimizers.__dict__[optimizer_name]
    else:
        # Raise an exception if the optimizer name is invalid
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

