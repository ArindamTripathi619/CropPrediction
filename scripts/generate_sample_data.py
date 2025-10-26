"""
Generate sample datasets for testing and demonstration purposes.
These are synthetic datasets that mimic real crop recommendation data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_crop_data(n_samples=2200):
    """Generate synthetic crop recommendation dataset."""
    np.random.seed(42)
    
    data = {
        'N': np.random.randint(20, 150, n_samples),
        'P': np.random.randint(10, 150, n_samples),
        'K': np.random.randint(20, 150, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples).round(1),
        'humidity': np.random.uniform(20, 90, n_samples).round(1),
        'ph': np.random.uniform(4.5, 9.0, n_samples).round(1),
        'rainfall': np.random.uniform(50, 300, n_samples).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels (crop types) based on conditions
    crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
        'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    
    labels = []
    for _, row in df.iterrows():
        # Simple rule-based labeling for demonstration
        if row['temperature'] > 28 and row['rainfall'] > 150:
            crop = np.random.choice(crops[:10])  # Rain-dependent crops
        elif row['N'] > 80 and row['P'] > 50:
            crop = np.random.choice(crops[10:15])  # Fruit crops
        elif row['temperature'] < 20:
            crop = np.random.choice(crops[15:18])  # Cool climate crops
        else:
            crop = np.random.choice(crops)
        labels.append(crop)
    
    df['label'] = labels
    return df


def generate_fertilizer_data(n_samples=2000):
    """Generate synthetic fertilizer recommendation dataset."""
    np.random.seed(42)
    
    data = {
        'Temparature': np.random.uniform(15, 35, n_samples).round(1),
        'Humidity': np.random.uniform(20, 90, n_samples).round(1),
        'Moisture': np.random.uniform(5, 30, n_samples).round(1),
        'Soil Type': np.random.choice(['Loamy', 'Sandy', 'Clayey', 'Black'], n_samples),
        'Crop Type': np.random.choice([
            'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
            'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses'
        ], n_samples),
        'Nitrogen': np.random.uniform(10, 100, n_samples).round(1),
        'Potassium': np.random.uniform(10, 100, n_samples).round(1),
        'Phosphorous': np.random.uniform(10, 100, n_samples).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Generate fertilizer type based on NPK values
    fertilizer_types = [
        'Urea', 'DAP', '14-35-14', '28-28', '17-17-17', 
        '20-20', '10-26-26', 'Urea', 'DAP'
    ]
    
    labels = []
    for _, row in df.iterrows():
        # Simple rule-based labeling
        if row['Nitrogen'] < 40:
            fert = 'Urea'
        elif row['Phosphorous'] < 30:
            fert = np.random.choice(['DAP', '14-35-14'])
        elif row['Potassium'] < 35:
            fert = np.random.choice(['17-17-17', '20-20', '10-26-26'])
        else:
            fert = np.random.choice(fertilizer_types)
        labels.append(fert)
    
    df['Fertilizer'] = labels
    return df


def generate_yield_data(n_samples=1500):
    """Generate synthetic crop yield dataset."""
    np.random.seed(42)
    
    states = ['Andhra Pradesh', 'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Gujarat']
    districts = ['District A', 'District B', 'District C', 'District D', 'District E']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    
    data = {
        'State': np.random.choice(states, n_samples),
        'District': np.random.choice(districts, n_samples),
        'Crop_Year': np.random.choice([2015, 2016, 2017, 2018, 2019, 2020], n_samples),
        'Season': np.random.choice(seasons, n_samples),
        'Crop': np.random.choice(crops, n_samples),
        'Area': np.random.uniform(1000, 50000, n_samples).round(1),
        'Production': np.random.uniform(5000, 200000, n_samples).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate rainfall (synthetic)
    df['Annual_Rainfall'] = np.random.uniform(400, 2000, n_samples).round(1)
    
    # Calculate yield (Production / Area)
    df['Yield'] = (df['Production'] / df['Area']).round(2)
    
    return df


def main():
    """Generate all sample datasets."""
    print("Generating sample datasets...")
    print("=" * 50)
    
    # Create data/raw directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save datasets
    print("\n1. Generating crop recommendation data...")
    crop_df = generate_crop_data()
    crop_path = raw_dir / "crop_data.csv"
    crop_df.to_csv(crop_path, index=False)
    print(f"Saved to {crop_path} ({len(crop_df)} samples)")
    
    print("\n2. Generating fertilizer recommendation data...")
    fert_df = generate_fertilizer_data()
    fert_path = raw_dir / "fertilizer_data.csv"
    fert_df.to_csv(fert_path, index=False)
    print(f"Saved to {fert_path} ({len(fert_df)} samples)")
    
    print("\n3. Generating yield estimation data...")
    yield_df = generate_yield_data()
    yield_path = raw_dir / "yield_data.csv"
    yield_df.to_csv(yield_path, index=False)
    print(f"Saved to {yield_path} ({len(yield_df)} samples)")
    
    print("\n" + "=" * 50)
    print("All sample datasets generated successfully!")
    print("\nNote: These are synthetic datasets for demonstration.")
    print("For actual production, download real datasets from Kaggle.")
    print("=" * 50)


if __name__ == "__main__":
    main()

