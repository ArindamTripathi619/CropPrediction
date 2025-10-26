"""
Unit tests for model classes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.crop_model import CropRecommendationModel


def test_crop_model():
    """Test crop recommendation model."""
    print("Testing Crop Recommendation Model...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 7
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 22, n_samples)  # 22 crop types
    
    # Convert to DataFrame
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = CropRecommendationModel(
        n_estimators=10,
        random_state=42
    )
    model.train(X_train, y_train)
    
    # Test predictions
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length mismatch"
    
    # Test evaluation
    metrics = model.evaluate(X_test, y_test)
    assert 'accuracy' in metrics, "Accuracy metric missing"
    
    print("✓ Crop model test passed")
    return True


def test_model_save_load():
    """Test model save and load functionality."""
    print("Testing model save/load...")
    
    # Create and train a simple model
    np.random.seed(42)
    X = np.random.rand(50, 7)
    y = np.random.randint(0, 22, 50)
    
    model = CropRecommendationModel(n_estimators=5, random_state=42)
    model.train(X, y)
    
    # Save model
    save_path = "tests/test_model.pkl"
    model.save_model(save_path)
    
    # Create new model and load
    new_model = CropRecommendationModel(n_estimators=5, random_state=42)
    new_model.load_model(save_path)
    
    # Test that predictions match
    predictions1 = model.predict(X[:5])
    predictions2 = new_model.predict(X[:5])
    
    assert np.array_equal(predictions1, predictions2), "Predictions don't match after load"
    
    print("✓ Model save/load test passed")
    return True


if __name__ == "__main__":
    print("Running tests...\n")
    
    try:
        test_crop_model()
        test_model_save_load()
        
        print("\n✓ All tests passed!")
    
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

