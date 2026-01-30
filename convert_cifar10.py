"""
Convert CIFAR-10 to ImageFolder format
"""

import os
from PIL import Image
from torchvision import datasets
import numpy as np

def convert_cifar10():
    """Convert CIFAR-10 to ImageFolder structure."""
    
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root='./cifar10_raw', train=True, download=False)
    test_dataset = datasets.CIFAR10(root='./cifar10_raw', train=False, download=False)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Found {len(train_dataset)} training images")
    print(f"Found {len(test_dataset)} test images")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(f'./data/{split}/{class_name}', exist_ok=True)
    
    print("\nConverting training data...")
    # Convert training data (80% train, 20% val)
    for idx, (img, label) in enumerate(train_dataset):
        class_name = class_names[label]
        
        # Split: 80% train, 20% val
        if idx < len(train_dataset) * 0.8:
            split = 'train'
        else:
            split = 'val'
        
        # Save image
        img_path = f'./data/{split}/{class_name}/{idx:05d}.png'
        img.save(img_path)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(train_dataset)} images...")
    
    print("\nConverting test data...")
    # Convert test data
    for idx, (img, label) in enumerate(test_dataset):
        class_name = class_names[label]
        img_path = f'./data/test/{class_name}/{idx:05d}.png'
        img.save(img_path)
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(test_dataset)} images...")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    
    # Print summary
    print("\nDataset structure:")
    for split in ['train', 'val', 'test']:
        total = 0
        for class_name in class_names:
            path = f'./data/{split}/{class_name}'
            count = len(os.listdir(path))
            total += count
        print(f"  {split:<6}: {total:>6} images")
    
    print("\nYou can now run: python train.py")

if __name__ == '__main__':
    convert_cifar10()
