"""
Image Classification System - Training Script
Fine-tuned ResNet-50 with Transfer Learning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


class ImageClassifier:
    """Main class for training and evaluating image classification models."""
    
    def __init__(self, num_classes, model_name='resnet50', device=None):
        """
        Initialize the image classifier.
        
        Args:
            num_classes (int): Number of output classes
            model_name (str): Base model to use ('resnet50', 'resnet101', etc.)
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _build_model(self):
        """Build transfer learning model with ResNet-50."""
        # Load pre-trained ResNet-50
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def unfreeze_layers(self, num_layers=10):
        """Unfreeze last N layers for fine-tuning."""
        # Get all parameters
        params = list(self.model.parameters())
        
        # Unfreeze last num_layers
        for param in params[-num_layers:]:
            param.requires_grad = True
    
    def prepare_data(self, train_dir, val_dir, batch_size=32):
        """
        Prepare data loaders with augmentation.
        
        Args:
            train_dir (str): Path to training data
            val_dir (str): Path to validation data
            batch_size (int): Batch size for training
        """
        # Data augmentation for training
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Store class names
        self.class_names = train_dataset.classes
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, epochs=25, lr=0.001, save_path='best_model.pth'):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            lr (float): Learning rate
            save_path (str): Path to save best model
        """
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_acc = 0.0
        
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print('-' * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }, save_path)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_acc*100:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_history(self, path='training_history.json'):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'num_classes': 10,  # CIFAR-10 has 10 classes
        'train_dir': './data/train',
        'val_dir': './data/val',
        'batch_size': 16,  # Start small for Mac
        'epochs': 5,       # Just 5 for testing
        'learning_rate': 0.001,
        'model_save_path': 'models/best_model.pth'
    }
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize classifier
    classifier = ImageClassifier(
        num_classes=config['num_classes'],
        model_name='resnet50'
    )
    
    # Prepare data
    print("Preparing data loaders...")
    classifier.prepare_data(
        config['train_dir'],
        config['val_dir'],
        batch_size=config['batch_size']
    )
    
    # Unfreeze last layers for fine-tuning
    print("Unfreezing layers for fine-tuning...")
    classifier.unfreeze_layers(num_layers=20)
    
    # Train model
    history = classifier.fit(
        epochs=config['epochs'],
        lr=config['learning_rate'],
        save_path=config['model_save_path']
    )
    
    # Save training history
    classifier.save_history('training_history.json')
    print("\nTraining history saved to 'training_history.json'")


if __name__ == '__main__':
    main()
