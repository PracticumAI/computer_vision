import requests
import os
import time
import tarfile
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential  # Import the Sequential model: a linear stack of layers from Keras module in TensorFlow.
from tensorflow.keras.layers import Dense  # Import the Dense layer: a fully connected neural network layer from Keras module in TensorFlow.
from tensorflow.keras.layers import Flatten  # Import the Flatten layer: used to convert input data into a 1D array from Keras module in TensorFlow.
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Import the SparseCategoricalCrossentropy loss function from Keras module in TensorFlow.
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers 
from tensorflow.keras import losses
from sklearn.metrics import confusion_matrix 

from sklearn.utils import class_weight


def load_display_data(path, batch_size=32, shape=(80,80,3), show_pictures=True,
                      stratify=True, return_cls_counts=False):
    '''Takes info on how to load and process images.
       Returns training and testing datasets. 
       If retern_cls_counts=True, this is also returned as 3rd value.
    '''
    print("***********************************************************************")
    print("Load data:")
    print(f"  - Loading the dataset from: {path}.")
    print(f"  - Using a batch size of: {batch_size}.")
    print(f"  - Resizing input images to: {shape}.")
    print(f"  - Stratify when sampling? {stratify}")
    print(f"  - Returning class counts to calculate wieghts? {return_cls_counts}")
    print("***********************************************************************")
    
    batch_size = batch_size  # Define the batch size
    
    # Define the image size using the 1st 2 elements of the shape parameter
    # We don't need the number of channels here, just the dimensions to use
    image_size = shape[:2]

    # Get the class names
    class_names = os.listdir(path)

    images = [] # Initialize the images list
    labels = [] # Initialize the labels list
    cls_counts={}
          
    # Get the images and labels to use for training and validation
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            images.append(image_path)
            labels.append(class_name)

    # Print the number of number of images per class
    print("\nFor the full dataset:")
    for class_name in class_names:
        print(f'Number of {class_name} images: {labels.count(class_name)}, or {labels.count(class_name)/len(labels)*100:.1f}%')
        # Save class count to return if requested
        cls_counts[class_name]=labels.count(class_name)
        
    # We originally used the image_dataset_from_directory function. That
    # doesn't suppot stratification. So, we're changing to use a dataframe
    # and an ImageDataGenerator.
          
    if stratify: # Use sklearn's train_test_split function to split the data 
                 # into training and testing sets
        # Split the data in a stratified manner
        X_train, X_val, y_train, y_val = train_test_split(images, labels, 
                                test_size=0.2, stratify=labels)
    else:
        # Split the data randomly
        X_train, X_val = train_test_split(images, test_size=0.2)

    # Build the DataFrames for the training and validation sets
    train_df = pd.DataFrame(list(zip(X_train, y_train)), columns=['image', 'class'])
    val_df = pd.DataFrame(list(zip(X_val, y_val)), columns=['image', 'class'])
    

    # Define the ImageDataGenerator class with rescaling for each channel
    # Normalizing the data is a good 1st step
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Define the training and validation data generators
    # Note that the training/validation split was already done above,
    # so, we are not using the validation_split here.
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


    if show_pictures:
        # Get the class names
        class_names = list(train_generator.class_indices.keys())
        print(class_names)

        # Display up to 3 images from each of the categories
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 10))
            
            # Get a single batch to use for display
            images, labels = train_generator.next()

            # Un-normalize the images for display
            images = images * 255

            # Filter images of the current class
            class_images = images[labels[:, i] == 1]
                
            # Number of images to show. We don't want to show more than 3 images.
            num_images = min(len(class_images), 3)
                
            for j in range(num_images):
                ax = plt.subplot(1, num_images, j + 1)
                plt.imshow(class_images[j].astype("uint8"))
                plt.title(class_name)
                plt.axis("off")
            plt.show()
    
    print("\nFor the training dataset:")         
    for class_name  in class_names:
        count = y_train.count(class_name)
        print(f'Number of {class_name} images: {count}, or {count/len(y_train)*100:.1f}%')

    if return_cls_counts:
        print(f"\nClass counts being returned: {cls_counts}.")
        return train_generator, val_generator, cls_counts
    else:
        return train_generator, val_generator

def load_optimizer(optimizer_name):
    '''Takes an optimizer name as a string and checks if it's valid'''

    # Check if the optimizer name is valid
    if optimizer_name in tf.keras.optimizers.__dict__:
        # Return the corresponding optimizer function
        return tf.keras.optimizers.__dict__[optimizer_name]
    else:
        # Raise an exception if the optimizer name is invalid
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

def make_model(activation='relu', shape=(80,80,3), num_classes=4):
    '''Sets up a model. 
          Takes in an activation function, shape for the input images, and number of classes.
          Returns the model.'''
    print("***********************************************************************")
    print("Make model:")
    print(f"  - Using the activation function: {activation}.")
    print(f"  - Model will have {num_classes} classes.")
    print("***********************************************************************")
    
    # Define the model
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation=activation, input_shape=shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model       
        
def compile_train_model(X_train, X_test, model,
                        loss='categorical_crossentropy',
                        optimizer='Adam', learning_rate=0.0001, epochs=10, cls_counts=False, log_name=None):
    '''Compiles and trains the model. 
          Takes in an X_train, X_test, model, loss function, optimizer, learning rate,
          epochs, if class weights should be used, and a list of callbacks.
          Returns the compiled model and training history.'''
    
    # Deal with class weights
    
    num_classes = len(list(X_train.class_indices.keys()))
    class_indices = range(num_classes)
    
    if not cls_counts:
        cls_wt = [1] * num_classes # Create a list of 1s as long as number of classes
        class_weights = dict(zip(class_indices, cls_wt))  
    
    else:
        # Calculate class weights to deal with imballance
        class_names = list(X_train.class_indices.keys())
        print(class_names)
        # Make a y from cls_counts
        y_vals = []
        for cls in list(X_train.class_indices.keys()):
            y_vals += [X_train.class_indices[cls]] * int(cls_counts[cls])

        cls_wt = class_weight.compute_class_weight('balanced', 
                                      classes=np.unique(y_vals), 
                                      y=y_vals)

        class_weights = dict(zip(class_indices, cls_wt))
     
    

    print("***********************************************************************")
    print("Compile and Train the model:")
    print(f"  - Using the loss function: {loss}.")
    print(f"  - Using the optimizer: {optimizer}.")
    print(f"  - Using learning rate of: {learning_rate}.")
    print(f"  - Running for {epochs} epochs.")
    print(f"   -Using class weights: {class_weights})")
    print("***********************************************************************")
    # Compile the model
    
    opt= load_optimizer(optimizer)(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = f"logs/fit_{log_name}_{epochs}_{time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    # Train the model
    history = model.fit(X_train, epochs=epochs, validation_data=X_test, class_weight=class_weights, callbacks=[tensorboard_callback])
    
    return model, history


def evaluate_model(train_set, val_set, model, history, num_classes=4):
    '''Evaluates a model. 
          Takes in a train_set, val_set, model, history, number of classes.'''
    print("***********************************************************************")
    print("Evaluate the model:")
    print("***********************************************************************")
    # Evaluate the model
    loss, accuracy = model.evaluate(train_set)
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

    class_names = list(train_set.class_indices.keys()) # Get the class names
    
    y_pred = np.argmax(model.predict(val_set), axis=-1) # Make predictions on the test set

    num_val_samples = val_set.samples  # total number of validation samples
    batch_size = val_set.batch_size  # batch size

    # Calculate the number of batches
    num_batches = np.ceil(num_val_samples / batch_size)
    num_batches = int(num_batches)

    # Get the true labels
    y_true = np.concatenate([y for x, y in (next(val_set) for _ in range(num_batches))], axis=0)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred)
    
    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(num_classes),class_names)
    plt.yticks(range(num_classes), class_names)
    # plt.colorbar()
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()