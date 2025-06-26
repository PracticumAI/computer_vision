import requests
import os
import time
import tarfile
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# For YOLOv8 Lightning wrapper
try:
    from ultralytics import YOLO
    import torch
    import yaml
except ImportError:
    print("Warning: ultralytics or torch not available. YOLO features will not work.")

def download_file(url="", filename="fruits_detection.zip"):

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
    
def manage_data(url="https://data.rc.ufl.edu/pub/practicum-ai/Computer_Vision/fruits_detection.tar.gz", filename="fruits_detection.tar.gz", folder_name='fruits_detection', dest='data'):
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

    print('Sorry, I cannot find the data. Please download it manually from https://data.rc.ufl.edu/pub/practicum-ai/Computer_Vision/fruits_detection.tar.gz and unpack it to the data folder.')      


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

class YOLOv8Lightning(pl.LightningModule):
    """PyTorch Lightning wrapper for YOLOv8 models"""
    
    def __init__(self, model_type='yolov8n', data_config=None, img_size=640, 
                 learning_rate=0.01, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize YOLO model
        self.yolo_model = YOLO(model_type + '.yaml')
        self.data_config = data_config
        self.img_size = img_size
        
        # Store training metrics for Lightning logging
        self.training_metrics = {}
        
    def forward(self, x):
        # YOLO forward pass
        return self.yolo_model(x)
    
    def training_step(self, batch, batch_idx):
        # YOLOv8 handles its own training step internally
        # We'll use this for logging purposes
        return None
    
    def validation_step(self, batch, batch_idx):
        # YOLOv8 handles its own validation internally
        return None
    
    def configure_optimizers(self):
        # YOLOv8 uses its own optimizer configuration
        # This is just for Lightning compatibility
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def train_yolo_model(data_config, model_type='yolov8n', max_epochs=10, img_size=640,
                     accelerator='auto', devices='auto', experiment_name='yolo_experiment',
                     learning_rate=0.01, weight_decay=0.0005, patience=5):
    """Train a YOLOv8 model using PyTorch Lightning wrapper
    
    Args:
        data_config: Path to YAML configuration file
        model_type: YOLO model variant (e.g., 'yolov8n', 'yolov8n-seg')
        max_epochs: Maximum number of training epochs
        img_size: Input image size
        accelerator: Device type ('auto', 'gpu', 'cpu')
        devices: Number/type of devices to use
        experiment_name: Name for the experiment logging
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
    
    Returns:
        tuple: (lightning_model, trainer, yolo_model)
    """
    
    print("="*60)
    print("YOLO LIGHTNING TRAINING SETUP")
    print("="*60)
    print(f"Model type: {model_type}")
    print(f"Data config: {data_config}")
    print(f"Image size: {img_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print("="*60)
    
    # Create Lightning wrapper
    lightning_model = YOLOv8Lightning(
        model_type=model_type,
        data_config=data_config,
        img_size=img_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create logger for TensorBoard
    logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=f'best-{experiment_name}-checkpoint',
        save_top_k=1,
        mode='min'
    )
    
    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    print("Starting YOLO training with Lightning integration...")
    
    # Train using YOLO's built-in training (more efficient than Lightning's loop for YOLO)
    yolo_model = YOLO(model_type + '.yaml')
    results = yolo_model.train(
        data=data_config,
        epochs=max_epochs,
        imgsz=img_size,
        lr0=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        project='lightning_logs',
        name=experiment_name,
        exist_ok=True
    )
    
    # Update the Lightning model with the trained YOLO model
    lightning_model.yolo_model = yolo_model
    
    print("Training completed successfully!")
    print(f"Results saved to: lightning_logs/{experiment_name}")
    
    return lightning_model, trainer, yolo_model

def evaluate_yolo_model(yolo_model, data_config, img_size=640):
    """Evaluate a trained YOLO model and display comprehensive metrics
    
    Args:
        yolo_model: Trained YOLO model
        data_config: Path to YAML configuration file
        img_size: Image size for evaluation
    
    Returns:
        dict: Evaluation results
    """
    
    print("="*60)
    print("YOLO MODEL EVALUATION")
    print("="*60)
    
    # Run validation
    results = yolo_model.val(
        data=data_config,
        imgsz=img_size,
        save_json=True,
        save_hybrid=True
    )
    
    # Extract key metrics
    metrics = {
        'mAP50': results.box.map50 if hasattr(results, 'box') else results.seg.map50,
        'mAP50-95': results.box.map if hasattr(results, 'box') else results.seg.map,
        'precision': results.box.mp if hasattr(results, 'box') else results.seg.mp,
        'recall': results.box.mr if hasattr(results, 'box') else results.seg.mr,
    }
    
    print("Evaluation Results:")
    print(f"  mAP@0.5: {metrics['mAP50']:.3f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print("="*60)
    
    return metrics

def compare_yolo_models(models_dict, data_config, img_size=640):
    """Compare multiple YOLO models and display results
    
    Args:
        models_dict: Dictionary of {model_name: yolo_model}
        data_config: Path to YAML configuration file
        img_size: Image size for evaluation
    
    Returns:
        dict: Comparison results
    """
    
    print("="*60)
    print("YOLO MODELS COMPARISON")
    print("="*60)
    
    comparison_results = {}
    
    for model_name, yolo_model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_yolo_model(yolo_model, data_config, img_size)
        comparison_results[model_name] = metrics
    
    # Display comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)
    
    for model_name, metrics in comparison_results.items():
        print(f"{model_name:<20} {metrics['mAP50']:<10.3f} {metrics['mAP50-95']:<15.3f} {metrics['precision']:<12.3f} {metrics['recall']:<10.3f}")
    
    return comparison_results

def visualize_yolo_results(yolo_model, test_images, conf_threshold=0.25):
    """Visualize YOLO model predictions on test images
    
    Args:
        yolo_model: Trained YOLO model
        test_images: List of image paths or single image path
        conf_threshold: Confidence threshold for predictions
    """
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    if isinstance(test_images, str):
        test_images = [test_images]
    
    # Run inference
    results = yolo_model.predict(test_images, conf=conf_threshold, save=False)
    
    # Display results
    fig, axes = plt.subplots(1, len(test_images), figsize=(5*len(test_images), 5))
    if len(test_images) == 1:
        axes = [axes]
    
    for i, (img_path, result) in enumerate(zip(test_images, results)):
        # Load original image
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Predictions: {os.path.basename(img_path)}")
        axes[i].axis('off')
        
        # Add predictions overlay (this would need to be customized based on detection/segmentation)
        if hasattr(result, 'boxes') and result.boxes is not None:
            # Detection boxes
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red', linewidth=2)
                axes[i].add_patch(rect)
                axes[i].text(x1, y1, f'{result.names[int(cls)]} {conf:.2f}', 
                           bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
    
    plt.tight_layout()
    plt.show()

def train_yolo_with_hyperparameter_tuning(data_config, model_type='yolov8n', 
                                         hyperparams_grid=None, max_epochs=10):
    """Train YOLO models with different hyperparameter combinations
    
    Args:
        data_config: Path to YAML configuration file
        model_type: YOLO model variant
        hyperparams_grid: Dictionary of hyperparameter options
        max_epochs: Maximum epochs for each training run
    
    Returns:
        dict: Results for each hyperparameter combination
    """
    
    if hyperparams_grid is None:
        hyperparams_grid = {
            'learning_rate': [0.01, 0.001],
            'weight_decay': [0.0005, 0.001],
            'img_size': [640],
        }
    
    results = {}
    
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Generate all combinations
    import itertools
    param_names = list(hyperparams_grid.keys())
    param_values = list(hyperparams_grid.values())
    
    for i, combination in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, combination))
        experiment_name = f"tune_{i}_{model_type}"
        
        print(f"\nTraining with: {params}")
        
        # Train model
        lightning_model, trainer, yolo_model = train_yolo_model(
            data_config=data_config,
            model_type=model_type,
            max_epochs=max_epochs,
            experiment_name=experiment_name,
            **params
        )
        
        # Evaluate
        metrics = evaluate_yolo_model(yolo_model, data_config, params.get('img_size', 640))
        
        results[experiment_name] = {
            'params': params,
            'metrics': metrics,
            'model': yolo_model
        }
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['metrics']['mAP50'])
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"Configuration: {best_config[1]['params']}")
    print(f"mAP@0.5: {best_config[1]['metrics']['mAP50']:.3f}")
    
    return results

