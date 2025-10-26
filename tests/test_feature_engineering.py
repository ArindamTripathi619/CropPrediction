"""
Unit tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    create_npk_ratios,
    create_temperature_ranges,
    create_ph_ranges,
    create_rainfall_features,
    create_interaction_features,
    engineer_features
)


@pytest.mark.unit
class TestFeatureEngineering:
    """Test suite for feature engineering functions."""
    
    def test_create_npk_ratios(self, sample_crop_data):
        """Test NPK ratio feature creation."""
        X = sample_crop_data[['N', 'P', 'K']]
        result = create_npk_ratios(X)
        
        # Check new features exist
        assert 'NP_ratio' in result.columns
        assert 'NK_ratio' in result.columns
        assert 'PK_ratio' in result.columns
        assert 'total_npk' in result.columns
        assert 'npk_balance' in result.columns
        
        # Check no NaN or infinity
        assert not result['NP_ratio'].isnull().any()
        assert not result['NK_ratio'].isnull().any()
        assert not np.isinf(result['NP_ratio']).any()
        
    def test_create_temperature_ranges(self, sample_crop_data):
        """Test temperature range feature creation."""
        X = sample_crop_data.drop('label', axis=1)
        result = create_temperature_ranges(X)
        
        # Check new features exist
        assert 'temp_low' in result.columns
        assert 'temp_medium' in result.columns
        assert 'temp_high' in result.columns
        assert 'temp_deviation' in result.columns
        
        # Check boolean nature
        assert result['temp_low'].dtype == bool
        assert result['temp_medium'].dtype == bool
        assert result['temp_high'].dtype == bool
        
        # Check logical consistency (only one should be True per row)
        for idx in range(len(result)):
            count = sum([
                result.loc[idx, 'temp_low'],
                result.loc[idx, 'temp_medium'],
                result.loc[idx, 'temp_high']
            ])
            assert count == 1, "Only one temperature range should be True"
            
    def test_create_ph_ranges(self, sample_crop_data):
        """Test pH range feature creation."""
        X = sample_crop_data.drop('label', axis=1)
        result = create_ph_ranges(X)
        
        # Check new features exist
        assert 'ph_acidic' in result.columns
        assert 'ph_neutral' in result.columns
        assert 'ph_alkaline' in result.columns
        assert 'ph_deviation' in result.columns
        
        # Check boolean nature
        assert result['ph_acidic'].dtype == bool
        assert result['ph_neutral'].dtype == bool
        assert result['ph_alkaline'].dtype == bool
        
    def test_create_rainfall_features(self, sample_crop_data):
        """Test rainfall feature creation."""
        X = sample_crop_data.drop('label', axis=1)
        result = create_rainfall_features(X)
        
        # Check new features exist
        assert 'rainfall_low' in result.columns
        assert 'rainfall_medium' in result.columns
        assert 'rainfall_high' in result.columns
        
        # Check boolean nature
        assert result['rainfall_low'].dtype == bool
        assert result['rainfall_medium'].dtype == bool
        assert result['rainfall_high'].dtype == bool
        
    def test_create_interaction_features(self, sample_crop_data):
        """Test interaction feature creation."""
        X = sample_crop_data.drop('label', axis=1)
        result = create_interaction_features(X)
        
        # Check interaction features exist
        if 'temperature' in X.columns and 'humidity' in X.columns:
            assert 'temp_humidity' in result.columns
            
        if 'rainfall' in X.columns and 'temperature' in X.columns:
            assert 'rainfall_temp' in result.columns
            
    def test_engineer_features_complete(self, sample_crop_data):
        """Test complete feature engineering pipeline."""
        X = sample_crop_data.drop('label', axis=1)
        result = engineer_features(X)
        
        # Should have more columns than original
        assert result.shape[1] > X.shape[1]
        
        # Original columns should still exist
        for col in X.columns:
            assert col in result.columns
            
        # Check no NaN introduced
        original_nulls = X.isnull().sum().sum()
        new_nulls = result.isnull().sum().sum()
        assert new_nulls == original_nulls, "Feature engineering shouldn't introduce NaNs"
        
    def test_engineer_features_with_flags(self, sample_crop_data):
        """Test feature engineering with different flag combinations."""
        X = sample_crop_data.drop('label', axis=1)
        
        # Only ratios
        result = engineer_features(X, include_ratios=True, include_ranges=False, include_interactions=False)
        assert 'NP_ratio' in result.columns
        assert 'temp_low' not in result.columns
        
        # Only ranges
        result = engineer_features(X, include_ratios=False, include_ranges=True, include_interactions=False)
        assert 'temp_low' in result.columns
        assert 'NP_ratio' not in result.columns
        
    def test_engineered_features_shape(self, engineered_features):
        """Test that engineered features have correct shape."""
        assert len(engineered_features) == 100  # From sample data
        assert engineered_features.shape[1] > 7  # More than original features
