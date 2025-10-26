"""
Training pipeline for all models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import load_crop_data, load_fertilizer_data, load_yield_data, split_data
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import engineer_features
from src.models.crop_model import CropRecommendationModel
from src.models.fertilizer_model import FertilizerPredictionModel
from src.models.yield_model import YieldEstimationModel
from src.utils.config import get_config


def train_crop_model(config):
    """Train crop recommendation model."""
    print("\n" + "="*50)
    print("Training Crop Recommendation Model")
    print("="*50)
    
    # Load data
    data_path = Path(config['data']['paths']['raw']) / "crop_data.csv"
    df = load_crop_data(str(data_path))
    
    if df is None:
        print("Crop data not found. Please download dataset first.")
        return None
    
    # Separate features and target
    target_col = 'label' if 'label' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Feature engineering
    X = engineer_features(X)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X, _ = preprocessor.preprocess_data(pd.concat([X, y], axis=1), target_col, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['data']['split_ratios']['train'],
        val_ratio=config['data']['split_ratios']['val'],
        random_state=config['data']['random_state']
    )
    
    # Train model
    crop_model = CropRecommendationModel(**config['models']['crop_recommendation']['params'])
    crop_model.train(X_train, y_train)
    
    # Evaluate
    train_metrics = crop_model.evaluate(X_train, y_train)
    val_metrics = crop_model.evaluate(X_val, y_val)
    test_metrics = crop_model.evaluate(X_test, y_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = Path(config['models']['crop_recommendation']['path'])
    crop_model.save_model(str(model_path))
    preprocessor.save_preprocessor(str(model_path.parent / 'preprocessor.pkl'))
    
    return crop_model


def train_fertilizer_model(config):
    """Train fertilizer prediction model."""
    print("\n" + "="*50)
    print("Training Fertilizer Prediction Model")
    print("="*50)
    
    # Load data
    data_path = Path(config['data']['paths']['raw']) / "fertilizer_data.csv"
    df = load_fertilizer_data(str(data_path))
    
    if df is None:
        print("Fertilizer data not found. Please download dataset first.")
        return None
    
    # Separate features and target
    target_col = 'Fertilizer' if 'Fertilizer' in df.columns else 'fertilizer' if 'fertilizer' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Feature engineering
    X = engineer_features(X)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X, _ = preprocessor.preprocess_data(pd.concat([X, y], axis=1), target_col, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['data']['split_ratios']['train'],
        val_ratio=config['data']['split_ratios']['val'],
        random_state=config['data']['random_state']
    )
    
    # Train model
    fertilizer_model = FertilizerPredictionModel(**config['models']['fertilizer_prediction']['params'])
    fertilizer_model.train(X_train, y_train)
    
    # Evaluate
    train_metrics = fertilizer_model.evaluate(X_train, y_train)
    val_metrics = fertilizer_model.evaluate(X_val, y_val)
    test_metrics = fertilizer_model.evaluate(X_test, y_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = Path(config['models']['fertilizer_prediction']['path'])
    fertilizer_model.save_model(str(model_path))
    preprocessor.save_preprocessor(str(model_path.parent / 'preprocessor.pkl'))
    
    return fertilizer_model


def train_yield_model(config):
    """Train yield estimation model."""
    print("\n" + "="*50)
    print("Training Yield Estimation Model")
    print("="*50)
    
    # Load data
    data_path = Path(config['data']['paths']['raw']) / "yield_data.csv"
    df = load_yield_data(str(data_path))
    
    if df is None:
        print("Yield data not found. Please download dataset first.")
        return None
    
    # Separate features and target
    target_col = 'Production' if 'Production' in df.columns else 'Yield' if 'Yield' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Feature engineering
    X = engineer_features(X)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X, _ = preprocessor.preprocess_data(pd.concat([X, y], axis=1), target_col, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['data']['split_ratios']['train'],
        val_ratio=config['data']['split_ratios']['val'],
        random_state=config['data']['random_state']
    )
    
    # Train model
    yield_model = YieldEstimationModel(**config['models']['yield_estimation']['params'])
    yield_model.train(X_train, y_train)
    
    # Evaluate
    train_metrics = yield_model.evaluate(X_train, y_train)
    val_metrics = yield_model.evaluate(X_val, y_val)
    test_metrics = yield_model.evaluate(X_test, y_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = Path(config['models']['yield_estimation']['path'])
    yield_model.save_model(str(model_path))
    preprocessor.save_preprocessor(str(model_path.parent / 'preprocessor.pkl'))
    
    return yield_model


def train_all_models():
    """Train all models."""
    config = get_config()
    
    # Train crop model
    crop_model = train_crop_model(config)
    
    # Train fertilizer model
    fertilizer_model = train_fertilizer_model(config)
    
    # Train yield model
    yield_model = train_yield_model(config)
    
    print("\n" + "="*50)
    print("All models trained successfully!")
    print("="*50)
    
    return crop_model, fertilizer_model, yield_model


if __name__ == "__main__":
    train_all_models()

