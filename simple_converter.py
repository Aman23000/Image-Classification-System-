"""
Simple CIFAR-10 to ImageFolder converter
"""

import os
from PIL import Image
from torchvision import datasets

print("Loading CIFAR-10...")

# Load datasets
train_data = datasets.CIFAR10(root='./cifar10_raw', train=True, download=False)
test_data = datasets.CIFAR10(root='./cifar10_raw', train=False, download=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Create directories
print("Creating directories...")
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(f'data/{split}/{cls}', exist_ok=True)

print(f"Converting {len(train_data)} training images...")

# Split training data: 80% train, 20% val
split_idx = int(len(train_data) * 0.8)

for idx, (img, label) in enumerate(train_data):
    cls = classes[label]
    split = 'train' if idx < split_idx else 'val'
    img.save(f'data/{split}/{cls}/{idx:05d}.png')
    
    if (idx + 1) % 5000 == 0:
        print(f"  {idx + 1}/{len(train_data)}...")

print(f"Converting {len(test_data)} test images...")

for idx, (img, label) in enumerate(test_data):
    cls = classes[label]
    img.save(f'data/test/{cls}/{idx:05d}.png')
    
    if (idx + 1) % 2000 == 0:
        print(f"  {idx + 1}/{len(test_data)}...")

print("\n✅ Done!")
print(f"Train: {split_idx} images")
print(f"Val: {len(train_data) - split_idx} images")
print(f"Test: {len(test_data)} images")
print("\nYou can now run: python train.py")
