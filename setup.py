"""
Setup script for Smart Crop Recommendation System.
Run this to initialize the project.
"""

import os
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "models/crop_recommendation",
        "models/fertilizer_prediction",
        "models/yield_estimation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_data_placeholder():
    """Create placeholder files in data/raw directory."""
    placeholders = [
        "data/raw/crop_data.csv",
        "data/raw/fertilizer_data.csv",
        "data/raw/yield_data.csv"
    ]
    
    for placeholder in placeholders:
        if not Path(placeholder).exists():
            with open(placeholder, 'w') as f:
                f.write("# Placeholder file\n")
                f.write("# Replace this file with the actual dataset from Kaggle\n")
            print(f"✓ Created placeholder: {placeholder}")


def main():
    """Main setup function."""
    print("Setting up Smart Crop Recommendation System...")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Create placeholder files
    print("\n2. Creating placeholder files...")
    create_data_placeholder()
    
    print("\n" + "=" * 50)
    print("✓ Setup complete!")
    print("\nNext steps:")
    print("1. Download datasets from Kaggle (see QUICKSTART.md)")
    print("2. Place CSV files in data/raw/ directory")
    print("3. Run: python -m src.training.train")
    print("4. Run: streamlit run app/app.py")
    print("\nFor detailed instructions, see QUICKSTART.md")
    print("=" * 50)


if __name__ == "__main__":
    main()

