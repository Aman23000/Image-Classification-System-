"""
Image Classification System - Evaluation and Visualization
Tools for model evaluation and result visualization
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path


class ModelEvaluator:
    """Evaluate and visualize model performance."""
    
    @staticmethod
    def plot_training_history(history_path='training_history.json', save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            history_path (str): Path to training history JSON
            save_path (str): Path to save plot
        """
        # Load history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        train_acc = [acc * 100 for acc in history['train_acc']]
        val_acc = [acc * 100 for acc in history['val_acc']]
        
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def evaluate_model(model, data_loader, class_names, device='cuda'):
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            data_loader: Test data loader
            class_names: List of class names
            device: Device to use
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        model = model.to(device)
        
        all_preds = []
        all_labels = []
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Classification report
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_per_class_metrics(report, save_path=None):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            report: Classification report dictionary
            save_path: Path to save plot
        """
        # Extract per-class metrics
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def generate_report(results, output_path='evaluation_report.txt'):
        """
        Generate text evaluation report.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
        """
        report = results['classification_report']
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("IMAGE CLASSIFICATION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['accuracy']*100:.2f}%\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 60 + "\n")
            
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"{class_name:<20} {metrics['precision']:<12.4f} "
                           f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f}\n")
            
            f.write("-" * 60 + "\n\n")
            
            f.write("Average Metrics:\n")
            f.write("-" * 60 + "\n")
            
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    f.write(f"{avg_type}:\n")
                    f.write(f"  Precision: {report[avg_type]['precision']:.4f}\n")
                    f.write(f"  Recall:    {report[avg_type]['recall']:.4f}\n")
                    f.write(f"  F1-Score:  {report[avg_type]['f1-score']:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"Evaluation report saved to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Evaluation and Visualization')
    parser.add_argument('--action', type=str, required=True,
                       choices=['plot-history', 'evaluate', 'all'],
                       help='Action to perform')
    parser.add_argument('--history', type=str, default='training_history.json',
                       help='Path to training history')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    evaluator = ModelEvaluator()
    
    if args.action in ['plot-history', 'all']:
        evaluator.plot_training_history(
            args.history,
            save_path=f'{args.output_dir}/training_history.png'
        )
    
    if args.action == 'all':
        print("\nFor full evaluation, please run evaluate with model and data")


if __name__ == '__main__':
    main()
