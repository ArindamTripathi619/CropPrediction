"""
Script to help download datasets from Kaggle.
Run this script to set up your datasets.

Note: You need to have Kaggle API credentials set up.
Place your kaggle.json file in ~/.kaggle/ directory.
"""

import os
import zipfile
from pathlib import Path

def download_datasets():
    """
    Download datasets from Kaggle.
    Make sure you have kaggle package installed: pip install kaggle
    And your credentials set up at ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print("Please install kaggle package: pip install kaggle")
        print("And set up your credentials at ~/.kaggle/kaggle.json")
        return
    
    # Create data/raw directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs from Kaggle
    datasets = {
        "crop_data": "atharvaingle/crop-recommendation-dataset",
        "fertilizer_data": "gdabhishek/fertilizer-prediction",
        "yield_data": "patelris/crop-yield-prediction-dataset"
    }
    
    print("Downloading datasets from Kaggle...")
    print("Make sure you have accepted the datasets on Kaggle first!\n")
    
    for name, dataset in datasets.items():
        print(f"Downloading {name}...")
        try:
            kaggle.api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)
            print(f"✓ {name} downloaded successfully\n")
        except Exception as e:
            print(f"✗ Error downloading {name}: {str(e)}\n")
            print("Tips:")
            print("1. Make sure you've accepted the dataset terms on Kaggle")
            print("2. Check your kaggle.json credentials")
            print("3. You can manually download from:")
            if name == "crop_data":
                print("   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
            elif name == "fertilizer_data":
                print("   https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction")
            elif name == "yield_data":
                print("   https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset")
            print()
    
    # Rename files to expected names
    print("Organizing downloaded files...")
    files_to_rename = {
        "Crop_recommendation.csv": "crop_data.csv",
        "fertilizer_recommendation.csv": "fertilizer_data.csv",
        "crop_yield.csv": "yield_data.csv"
    }
    
    for old_name, new_name in files_to_rename.items():
        old_path = raw_dir / old_name
        new_path = raw_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)
            print(f"✓ Renamed {old_name} to {new_name}")
    
    print("\n✓ Dataset download complete!")


if __name__ == "__main__":
    download_datasets()

