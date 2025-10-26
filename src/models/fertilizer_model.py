"""
Fertilizer prediction model implementation.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .base_model import BaseModel
import numpy as np


class FertilizerPredictionModel(BaseModel):
    """
    Model for predicting fertilizer type and dosage.
    """
    
    def __init__(self, algorithm='RandomForest', task='classification', **kwargs):
        """
        Initialize fertilizer prediction model.
        
        Args:
            algorithm: Model algorithm ('RandomForest', 'DecisionTree')
            task: Task type ('classification' or 'regression')
            **kwargs: Model hyperparameters
        """
        super().__init__(f"FertilizerPrediction_{algorithm}")
        self.algorithm = algorithm
        self.task = task
        self.params = kwargs
        
    def build_model(self):
        """Build the fertilizer prediction model."""
        if self.algorithm == 'RandomForest':
            if self.task == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=self.params.get('n_estimators', 100),
                    max_depth=self.params.get('max_depth', 15),
                    random_state=self.params.get('random_state', 42),
                    n_jobs=-1
                )
            else:  # regression
                self.model = RandomForestRegressor(
                    n_estimators=self.params.get('n_estimators', 100),
                    max_depth=self.params.get('max_depth', 15),
                    random_state=self.params.get('random_state', 42),
                    n_jobs=-1
                )
        elif self.algorithm == 'DecisionTree':
            if self.task == 'classification':
                self.model = DecisionTreeClassifier(
                    max_depth=self.params.get('max_depth', 20),
                    random_state=self.params.get('random_state', 42)
                )
            else:  # regression
                self.model = DecisionTreeRegressor(
                    max_depth=self.params.get('max_depth', 20),
                    random_state=self.params.get('random_state', 42)
                )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def predict_fertilizer_type(self, X):
        """
        Predict fertilizer type (classification).
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted fertilizer type
        """
        return self.predict(X)
    
    def predict_npk_dosage(self, X):
        """
        Predict NPK dosage requirements (regression).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of NPK dosage predictions [N, P, K]
        """
        if self.task != 'regression':
            raise ValueError("Model is not configured for regression task.")
        return self.predict(X)
    
    def get_fertilizer_recommendations(self, X, top_k=3):
        """
        Get top fertilizer recommendations with confidence.
        
        Args:
            X: Feature matrix
            top_k: Number of recommendations
            
        Returns:
            List of fertilizer recommendations
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        if self.task == 'classification' and hasattr(self.model, 'predict_proba'):
            # Get probabilities for fertilizer types
            proba = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            
            # Get top K
            top_indices = np.argsort(proba)[::-1][:top_k]
            recommendations = [(classes[i], proba[i]) for i in top_indices]
            
            return recommendations
        else:
            # For regression or models without proba
            prediction = self.predict(X)
            return [(prediction[0], 1.0)]

