"""
Crop recommendation model implementation.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel
import numpy as np


class CropRecommendationModel(BaseModel):
    """
    Model for recommending crops based on soil and environmental parameters.
    """
    
    def __init__(self, algorithm='RandomForest', **kwargs):
        """
        Initialize crop recommendation model.
        
        Args:
            algorithm: Model algorithm ('RandomForest', 'DecisionTree')
            **kwargs: Model hyperparameters
        """
        super().__init__(f"CropRecommendation_{algorithm}")
        self.algorithm = algorithm
        self.params = kwargs
        
    def build_model(self):
        """Build the crop recommendation model."""
        if self.algorithm == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 15),
                random_state=self.params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.algorithm == 'DecisionTree':
            self.model = DecisionTreeClassifier(
                max_depth=self.params.get('max_depth', 20),
                random_state=self.params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def get_top_predictions(self, X, top_k=3):
        """
        Get top K crop recommendations with confidence scores.
        
        Args:
            X: Feature matrix
            top_k: Number of top recommendations
            
        Returns:
            List of tuples (crop, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model does not have class information.")
        
        # Get class probabilities
        proba = self.model.predict_proba(X)[0]
        
        # Get top K classes
        top_indices = np.argsort(proba)[::-1][:top_k]
        
        # Get crop names and confidence scores
        crop_classes = self.model.classes_
        recommendations = [(crop_classes[i], proba[i]) for i in top_indices]
        
        return recommendations
    
    def get_confidence_by_class(self, X):
        """
        Get confidence scores for all classes.
        
        Args:
            X: Feature matrix
            
        Returns:
            dict: Dictionary mapping crop names to confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model does not have class information.")
        
        proba = self.model.predict_proba(X)[0]
        crop_classes = self.model.classes_
        
        return dict(zip(crop_classes, proba))

