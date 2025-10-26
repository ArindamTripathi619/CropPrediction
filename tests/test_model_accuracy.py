"""
Model accuracy and performance tests.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.crop_model import CropRecommendationModel
from src.models.fertilizer_model import FertilizerPredictionModel
from src.models.yield_model import YieldEstimationModel


@pytest.mark.model
class TestCropModelAccuracy:
    """Test crop recommendation model accuracy."""
    
    def test_model_trains_successfully(self, sample_crop_data):
        """Test that model trains without errors."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        model = CropRecommendationModel(n_estimators=50, random_state=42)
        model.train(X, y)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_model_accuracy_threshold(self, sample_crop_data):
        """Test that model meets minimum accuracy threshold."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        model = CropRecommendationModel(n_estimators=50, random_state=42)
        model.train(X, y)
        
        metrics = model.evaluate(X, y)
        
        # Model should achieve at least 20% accuracy on training data (very conservative)
        assert metrics['accuracy'] > 0.2, f"Accuracy too low: {metrics['accuracy']}"
        
    def test_model_predictions_valid(self, trained_crop_model, sample_crop_data):
        """Test that predictions are valid crop labels."""
        X = sample_crop_data.drop('label', axis=1).iloc[:10]
        
        predictions = trained_crop_model.predict(X)
        
        # Check predictions are from valid classes
        assert len(predictions) == len(X)
        assert all(pred in trained_crop_model.model.classes_ for pred in predictions)
        
    def test_model_confidence_scores(self, trained_crop_model, sample_crop_data):
        """Test that confidence scores are valid probabilities."""
        X = sample_crop_data.drop('label', axis=1).iloc[:5]
        
        # Get top predictions with confidence
        recommendations = trained_crop_model.get_top_predictions(X, top_k=3)
        
        # Check confidence scores
        for crop, confidence in recommendations:
            assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
            
        # Check top predictions are sorted by confidence
        confidences = [conf for _, conf in recommendations]
        assert confidences == sorted(confidences, reverse=True), "Predictions should be sorted by confidence"
        
    def test_cross_validation_score(self, sample_crop_data):
        """Test cross-validation performance."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        model = CropRecommendationModel(n_estimators=30, random_state=42)
        model.build_model()
        
        # 3-fold cross-validation
        scores = cross_val_score(model.model, X, y, cv=3, scoring='accuracy')
        
        # Mean accuracy should be reasonable
        mean_accuracy = scores.mean()
        assert mean_accuracy > 0.15, f"Cross-validation accuracy too low: {mean_accuracy}"
        
    def test_model_consistency(self, sample_crop_data):
        """Test that model produces consistent results."""
        X = sample_crop_data.drop('label', axis=1)
        y = sample_crop_data['label']
        
        # Train two models with same parameters
        model1 = CropRecommendationModel(n_estimators=20, random_state=42)
        model2 = CropRecommendationModel(n_estimators=20, random_state=42)
        
        model1.train(X, y)
        model2.train(X, y)
        
        # Predictions should be identical
        X_test = X.iloc[:10]
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        assert np.array_equal(pred1, pred2), "Models with same seed should produce identical predictions"


@pytest.mark.model
class TestFertilizerModelAccuracy:
    """Test fertilizer prediction model accuracy."""
    
    def test_fertilizer_model_trains(self, sample_fertilizer_data):
        """Test fertilizer model training."""
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        X = sample_fertilizer_data.drop('Fertilizer', axis=1).copy()
        y = sample_fertilizer_data['Fertilizer']
        
        # Encode categorical columns
        le_soil = LabelEncoder()
        le_crop = LabelEncoder()
        X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
        X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])
        
        model = FertilizerPredictionModel(n_estimators=30, random_state=42)
        model.train(X, y)
        
        assert model.is_trained
        
        # Check accuracy
        metrics = model.evaluate(X, y)
        assert metrics['accuracy'] > 0.2
        
    def test_fertilizer_predictions_valid(self, sample_fertilizer_data):
        """Test that fertilizer predictions are valid."""
        from sklearn.preprocessing import LabelEncoder
        
        X = sample_fertilizer_data.drop('Fertilizer', axis=1).copy()
        y = sample_fertilizer_data['Fertilizer']
        
        le_soil = LabelEncoder()
        le_crop = LabelEncoder()
        X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
        X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])
        
        model = FertilizerPredictionModel(n_estimators=20, random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X.iloc[:10])
        
        # Predictions should be valid fertilizer types
        assert len(predictions) == 10
        assert all(pred in model.model.classes_ for pred in predictions)


@pytest.mark.model
class TestYieldModelAccuracy:
    """Test yield estimation model accuracy."""
    
    def test_yield_model_trains(self, sample_yield_data):
        """Test yield model training."""
        from sklearn.preprocessing import LabelEncoder
        
        X = sample_yield_data.drop('Production', axis=1).copy()
        y = sample_yield_data['Production']
        
        # Encode categorical columns
        for col in ['State', 'District', 'Season', 'Crop']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        model = YieldEstimationModel(n_estimators=30, random_state=42)
        model.train(X, y)
        
        assert model.is_trained
        
        # Check R² score
        metrics = model.evaluate(X, y)
        assert 'r2_score' in metrics
        
    def test_yield_predictions_non_negative(self, sample_yield_data):
        """Test that yield predictions are non-negative."""
        from sklearn.preprocessing import LabelEncoder
        
        X = sample_yield_data.drop('Production', axis=1).copy()
        y = sample_yield_data['Production']
        
        for col in ['State', 'District', 'Season', 'Crop']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        model = YieldEstimationModel(n_estimators=20, random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X.iloc[:10])
        
        # All predictions should be non-negative
        assert all(pred >= 0 for pred in predictions), "Yield predictions should be non-negative"
        
    def test_yield_model_reasonable_range(self, sample_yield_data):
        """Test that yield predictions are in reasonable range."""
        from sklearn.preprocessing import LabelEncoder
        
        X = sample_yield_data.drop('Production', axis=1).copy()
        y = sample_yield_data['Production']
        
        for col in ['State', 'District', 'Season', 'Crop']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        model = YieldEstimationModel(n_estimators=20, random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X.iloc[:10])
        
        # Predictions should be within some multiple of actual values
        y_test = y.iloc[:10].values
        max_actual = y_test.max() * 10  # Allow 10x maximum as upper bound
        
        assert all(pred < max_actual for pred in predictions), "Predictions seem unreasonably high"


@pytest.mark.model
@pytest.mark.slow
class TestModelPerformanceMetrics:
    """Comprehensive model performance tests."""
    
    def test_trained_crop_model_on_real_data(self):
        """Test trained crop model performance if available."""
        import os
        from pathlib import Path
        
        model_path = Path("models/crop_recommendation/rf_model.pkl")
        
        if not model_path.exists():
            pytest.skip("Trained model not found")
            
        # Load trained model
        model = CropRecommendationModel()
        model.load_model(str(model_path))
        
        assert model.is_trained
        assert hasattr(model.model, 'classes_')
        
        # Model should have been trained on multiple crop classes
        assert len(model.model.classes_) > 5, "Model should recognize multiple crop types"
