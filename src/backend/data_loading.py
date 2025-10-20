"""
Data loading and preprocessing utilities for BloodMNIST dataset.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from medmnist import BloodMNIST


def get_data_transforms():
    """
    Define the transformations to be applied to the images.
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    # ToTensor() converts images to PyTorch Tensors and scales pixel values to [0, 1].
    # Normalize() adjusts the pixel values to have a mean of 0.5 and a standard
    # deviation of 0.5 across all channels. This helps with model training.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])  # Normalizes to [-1, 1] range
    ])
    return data_transform


def load_blood_mnist_datasets(data_transform, download=True):
    """
    Download and load the BloodMNIST dataset splits.
    
    Args:
        data_transform: Transformations to apply to images
        download (bool): Whether to download the dataset if not found locally
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # The MedMNIST library downloads the dataset and provides it in pre-defined splits.
    train_dataset = BloodMNIST(split="train", transform=data_transform, download=download)
    val_dataset = BloodMNIST(split="val", transform=data_transform, download=download)
    test_dataset = BloodMNIST(split="test", transform=data_transform, download=download)
    
    return train_dataset, val_dataset, test_dataset


def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """
    Print information about the dataset splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    print("\nDataset Information:")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Task: {train_dataset.info['task']}")
    print(f"Number of channels: {train_dataset.info['n_channels']}")
    print(f"Number of classes: {len(train_dataset.info['label'])}")
    print(f"Class labels: {train_dataset.info['label']}")


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128):
    """
    Create DataLoaders for training, validation, and test datasets.
    
    DataLoaders provide an iterable over the dataset, with support for batching,
    shuffling, and parallel data loading.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Shuffle the training data to ensure the model sees data in a random order
    # each epoch, which helps prevent overfitting.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # No need to shuffle validation or test data.
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\nCreated DataLoaders with batch size {batch_size}.")
    
    return train_loader, val_loader, test_loader


def show_image_batch(data_loader, dataset, title="Image Batch", nrow=16):
    """
    Visualizes a single batch of images from a DataLoader.
    
    Args:
        data_loader: DataLoader to get images from
        dataset: Dataset object (used to access label information)
        title (str): Title for the plot
        nrow (int): Number of images to display per row
    """
    # Get one batch of images and labels
    images, labels = next(iter(data_loader))
    
    # Create a grid of images
    # The `nrow` argument specifies how many images to display in each row.
    img_grid = torchvision.utils.make_grid(images, nrow=nrow)
    
    # The ToTensor transform moves the channel axis to the front (C, H, W).
    # Matplotlib expects the channel axis at the back (H, W, C).
    # We use permute() to rearrange the axes.
    # We also need to un-normalize the images to see them correctly.
    img_grid = img_grid.permute(1, 2, 0) * 0.5 + 0.5  # Un-normalize to [0, 1]
    img_grid = np.clip(img_grid.numpy(), 0, 1)  # Clip values to be safe

    # Get the class names for the labels in this batch
    class_names = [dataset.info['label'][str(l.item())] for l in labels]

    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.imshow(img_grid)
    plt.axis('off')
    
    # Print the first row of labels to give context to the image grid
    print("\nLabels for the first row of images:")
    print(", ".join(class_names[:nrow]))
    
    plt.show()
