"""
Image Classification System - Inference Script
Predict on new images using trained model
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np


class ImagePredictor:
    """Class for making predictions on new images."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to saved model checkpoint
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get class names
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        
        # Build model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
    
    def _build_model(self):
        """Build model architecture."""
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        return model
    
    def predict(self, image_path, top_k=3):
        """
        Predict class for a single image.
        
        Args:
            image_path (str): Path to image
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Predictions with probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = {
            'predictions': []
        }
        
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results['predictions'].append({
                'class': self.class_names[idx.item()],
                'probability': prob.item(),
                'confidence': f'{prob.item()*100:.2f}%'
            })
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of predictions
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image'] = image_path
            results.append(result)
        return results


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ImagePredictor(args.model)
    
    # Make prediction
    print(f"\nPredicting for: {args.image}")
    results = predictor.predict(args.image, top_k=args.top_k)
    
    # Print results
    print("\nTop Predictions:")
    print("-" * 50)
    for i, pred in enumerate(results['predictions'], 1):
        print(f"{i}. {pred['class']:<20} {pred['confidence']:>10}")
    print("-" * 50)


if __name__ == '__main__':
    main()
