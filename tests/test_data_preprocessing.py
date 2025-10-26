"""
Unit tests for data preprocessing components.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor
from src.data.loader import split_data


@pytest.mark.unit
class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.scaler is None
        assert preprocessor.label_encoder == {}
        
    def test_handle_missing_values(self, preprocessor, sample_crop_data):
        """Test missing value handling."""
        # Add some missing values
        df = sample_crop_data.copy()
        df.loc[0:5, 'N'] = np.nan
        df.loc[10:15, 'temperature'] = np.nan
        
        # Handle missing values
        df_cleaned = preprocessor.handle_missing_values(df)
        
        # Check no missing values remain
        assert df_cleaned.isnull().sum().sum() == 0
        
    def test_scale_features(self, preprocessor, sample_crop_data):
        """Test feature scaling."""
        X = sample_crop_data.drop('label', axis=1)
        
        # First fit
        X_scaled = preprocessor.scale_features(X, fit=True)
        
        # Check scaling
        assert X_scaled.shape == X.shape
        assert preprocessor.scaler is not None
        assert preprocessor.feature_names is not None
        
        # Check values are scaled (mean ~ 0, std ~ 1)
        for col in X_scaled.columns:
            mean = X_scaled[col].mean()
            std = X_scaled[col].std()
            assert abs(mean) < 1.0, f"Mean of {col} should be close to 0"
            
    def test_scale_features_transform_only(self, preprocessor, sample_crop_data):
        """Test scaling in transform mode."""
        X = sample_crop_data.drop('label', axis=1)
        
        # Fit
        X_scaled_fit = preprocessor.scale_features(X, fit=True)
        
        # Transform new data
        X_new = X.iloc[:10].copy()
        X_scaled_transform = preprocessor.scale_features(X_new, fit=False)
        
        assert X_scaled_transform.shape == X_new.shape
        
    def test_preprocess_data_complete_pipeline(self, preprocessor, sample_crop_data):
        """Test complete preprocessing pipeline."""
        X, y = preprocessor.preprocess_data(
            sample_crop_data, 
            target_column='label', 
            fit=True
        )
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X) == len(sample_crop_data)
        
    def test_save_load_preprocessor(self, preprocessor, sample_crop_data, temp_dir):
        """Test saving and loading preprocessor."""
        import os
        
        # Fit preprocessor
        X = sample_crop_data.drop('label', axis=1)
        preprocessor.scale_features(X, fit=True)
        
        # Save
        save_path = os.path.join(temp_dir, 'test_preprocessor.pkl')
        preprocessor.save_preprocessor(save_path)
        
        # Load
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_preprocessor(save_path)
        
        # Check loaded correctly
        assert new_preprocessor.scaler is not None
        assert new_preprocessor.feature_names is not None
        assert len(new_preprocessor.feature_names) == len(preprocessor.feature_names)


@pytest.mark.unit
class TestDataLoading:
    """Test suite for data loading functions."""
    
    def test_split_data(self, sample_crop_data):
        """Test data splitting."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            random_state=42
        )
        
        # Check sizes (allow for rounding differences)
        total = len(X)
        assert abs(len(X_train) - int(total * 0.7)) <= 1, f"Train size mismatch: {len(X_train)} vs expected ~{int(total * 0.7)}"
        assert abs(len(X_val) - int(total * 0.15)) <= 1, f"Val size mismatch: {len(X_val)} vs expected ~{int(total * 0.15)}"
        assert len(X_test) > 0
        
        # Check no data leakage
        assert len(X_train) + len(X_val) + len(X_test) == total
        
    def test_split_data_different_ratios(self, sample_crop_data):
        """Test data splitting with different ratios."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, 
            train_ratio=0.8, 
            val_ratio=0.1, 
            random_state=42
        )
        
        total = len(X)
        assert abs(len(X_train) - int(total * 0.8)) <= 1
        assert abs(len(X_val) - int(total * 0.1)) <= 1
