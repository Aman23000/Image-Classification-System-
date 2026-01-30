"""
Image Classification System - Data Utilities
Helper functions for data preparation and augmentation
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np


class DataPreparation:
    """Utility class for preparing image datasets."""
    
    @staticmethod
    def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split dataset into train/val/test sets.
        
        Args:
            source_dir (str): Directory with class folders
            output_dir (str): Output directory for splits
            train_ratio (float): Training data ratio
            val_ratio (float): Validation data ratio
            test_ratio (float): Test data ratio
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
        # Get all class folders
        source_path = Path(source_dir)
        classes = [d for d in source_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(classes)} classes")
        print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        
        total_images = 0
        
        for class_dir in classes:
            class_name = class_dir.name
            print(f"\nProcessing class: {class_name}")
            
            # Get all images in class
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            # Shuffle images
            random.shuffle(image_files)
            
            num_images = len(image_files)
            train_split = int(num_images * train_ratio)
            val_split = int(num_images * (train_ratio + val_ratio))
            
            # Split images
            train_images = image_files[:train_split]
            val_images = image_files[train_split:val_split]
            test_images = image_files[val_split:]
            
            print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
            
            # Copy images to respective folders
            for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                
                for img_path in images:
                    dest_path = os.path.join(split_class_dir, img_path.name)
                    shutil.copy2(img_path, dest_path)
            
            total_images += num_images
        
        print(f"\nDataset preparation complete!")
        print(f"Total images processed: {total_images}")
    
    @staticmethod
    def analyze_dataset(data_dir):
        """
        Analyze dataset statistics.
        
        Args:
            data_dir (str): Dataset directory
        """
        data_path = Path(data_dir)
        
        print(f"\nDataset Analysis: {data_dir}")
        print("=" * 60)
        
        total_images = 0
        class_stats = {}
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            if not split_path.exists():
                continue
            
            print(f"\n{split.upper()} Split:")
            print("-" * 60)
            
            classes = [d for d in split_path.iterdir() if d.is_dir()]
            split_total = 0
            
            for class_dir in sorted(classes):
                class_name = class_dir.name
                images = list(class_dir.glob('*.jpg')) + \
                        list(class_dir.glob('*.jpeg')) + \
                        list(class_dir.glob('*.png'))
                num_images = len(images)
                split_total += num_images
                
                if class_name not in class_stats:
                    class_stats[class_name] = {'train': 0, 'val': 0, 'test': 0}
                class_stats[class_name][split] = num_images
                
                print(f"  {class_name:<20}: {num_images:>6} images")
            
            print(f"  {'Total':<20}: {split_total:>6} images")
            total_images += split_total
        
        print(f"\n{'='*60}")
        print(f"Total images across all splits: {total_images}")
        print(f"Number of classes: {len(class_stats)}")
        
        # Check class balance
        print(f"\nClass Balance:")
        print("-" * 60)
        for class_name, stats in sorted(class_stats.items()):
            total_class = stats['train'] + stats['val'] + stats['test']
            print(f"  {class_name:<20}: {total_class:>6} ({total_class/total_images*100:.1f}%)")
    
    @staticmethod
    def check_image_quality(data_dir, min_size=224):
        """
        Check for corrupted or low-quality images.
        
        Args:
            data_dir (str): Dataset directory
            min_size (int): Minimum image dimension
        """
        print(f"\nChecking image quality...")
        print(f"Minimum size: {min_size}x{min_size}")
        
        data_path = Path(data_dir)
        issues = []
        
        # Check all images
        for img_path in data_path.rglob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    # Check size
                    if width < min_size or height < min_size:
                        issues.append(f"{img_path}: Size too small ({width}x{height})")
                    
                    # Try to load image data
                    img.verify()
                    
                except Exception as e:
                    issues.append(f"{img_path}: Error - {str(e)}")
        
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ All images passed quality check!")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preparation Utilities')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['split', 'analyze', 'check'],
                       help='Action to perform')
    parser.add_argument('--source', type=str, help='Source directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio')
    
    args = parser.parse_args()
    
    prep = DataPreparation()
    
    if args.action == 'split':
        prep.split_dataset(
            args.source,
            args.output,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    elif args.action == 'analyze':
        prep.analyze_dataset(args.source)
    elif args.action == 'check':
        prep.check_image_quality(args.source)


if __name__ == '__main__':
    main()
