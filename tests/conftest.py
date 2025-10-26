"""
Pytest fixtures and shared test utilities.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.crop_model import CropRecommendationModel
from src.models.fertilizer_model import FertilizerPredictionModel
from src.models.yield_model import YieldEstimationModel
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import engineer_features


@pytest.fixture
def sample_crop_data():
    """Generate sample crop recommendation data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8, 44, n_samples),
        'humidity': np.random.uniform(14, 100, n_samples),
        'ph': np.random.uniform(3.5, 9.9, n_samples),
        'rainfall': np.random.uniform(20, 300, n_samples),
        'label': np.random.choice(['rice', 'wheat', 'maize', 'cotton'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_fertilizer_data():
    """Generate sample fertilizer data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Temparature': np.random.uniform(15, 45, n_samples),
        'Humidity': np.random.uniform(20, 100, n_samples),
        'Moisture': np.random.uniform(10, 90, n_samples),
        'Soil Type': np.random.choice(['Loamy', 'Clayey', 'Black', 'Sandy'], n_samples),
        'Crop Type': np.random.choice(['Maize', 'Wheat', 'Rice', 'Cotton'], n_samples),
        'Nitrogen': np.random.randint(0, 100, n_samples),
        'Potassium': np.random.randint(0, 100, n_samples),
        'Phosphorous': np.random.randint(0, 100, n_samples),
        'Fertilizer': np.random.choice(['Urea', 'DAP', '28-28'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_yield_data():
    """Generate sample yield data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'State': np.random.choice(['Tamil Nadu', 'Gujarat', 'Maharashtra'], n_samples),
        'District': np.random.choice(['District A', 'District B', 'District C'], n_samples),
        'Crop_Year': np.random.randint(2015, 2021, n_samples),
        'Season': np.random.choice(['Kharif', 'Rabi', 'Zaid'], n_samples),
        'Crop': np.random.choice(['Rice', 'Wheat', 'Maize'], n_samples),
        'Area': np.random.uniform(100, 10000, n_samples),
        'Annual_Rainfall': np.random.uniform(500, 2500, n_samples),
        'Yield': np.random.uniform(1, 5, n_samples),
        'Production': np.random.uniform(100, 50000, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_crop_model(sample_crop_data):
    """Create and train a crop model."""
    X = sample_crop_data.drop('label', axis=1)
    y = sample_crop_data['label']
    
    model = CropRecommendationModel(n_estimators=10, random_state=42)
    model.train(X, y)
    
    return model


@pytest.fixture
def preprocessor():
    """Create a data preprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_input_features():
    """Sample input features for prediction."""
    return {
        'N': 50,
        'P': 40,
        'K': 40,
        'temperature': 25.0,
        'humidity': 80.0,
        'ph': 7.0,
        'rainfall': 200.0
    }


@pytest.fixture
def sample_fertilizer_input():
    """Sample fertilizer input."""
    return {
        'Temparature': 25.0,
        'Humidity': 80.0,
        'Moisture': 50.0,
        'Soil Type': 'Loamy',
        'Crop Type': 'Maize',
        'Nitrogen': 50,
        'Potassium': 40,
        'Phosphorous': 40
    }


@pytest.fixture
def engineered_features(sample_crop_data):
    """Generate engineered features from sample data."""
    X = sample_crop_data.drop('label', axis=1)
    return engineer_features(X)


# Helper functions for tests
def assert_model_trained(model):
    """Assert that a model is trained."""
    assert model.is_trained, "Model should be trained"
    assert model.model is not None, "Model should have an underlying ML model"


def assert_predictions_valid(predictions, expected_length):
    """Assert that predictions are valid."""
    assert len(predictions) == expected_length, f"Expected {expected_length} predictions"
    assert predictions is not None, "Predictions should not be None"
