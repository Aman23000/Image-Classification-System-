from torchvision import datasets
import os
print("Downloading CIFAR-10...")
datasets.CIFAR10(root='./cifar10_raw', train=True, download=True)
datasets.CIFAR10(root='./cifar10_raw', train=False, download=True)
print("Download complete!")
