"""
Yield estimation model implementation.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from .base_model import BaseModel
import numpy as np


class YieldEstimationModel(BaseModel):
    """
    Model for estimating crop yield based on various parameters.
    """
    
    def __init__(self, algorithm='RandomForest', **kwargs):
        """
        Initialize yield estimation model.
        
        Args:
            algorithm: Model algorithm ('RandomForest', 'DecisionTree')
            **kwargs: Model hyperparameters
        """
        super().__init__(f"YieldEstimation_{algorithm}")
        self.algorithm = algorithm
        self.params = kwargs
        
    def build_model(self):
        """Build the yield estimation model."""
        if self.algorithm == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 15),
                random_state=self.params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.algorithm == 'DecisionTree':
            self.model = DecisionTreeRegressor(
                max_depth=self.params.get('max_depth', 20),
                random_state=self.params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def predict_yield(self, X):
        """
        Predict crop yield.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted yield (kg/hectare)
        """
        predictions = self.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_yield_range(self, X, confidence_level=0.95):
        """
        Get yield prediction with confidence interval.
        
        Args:
            X: Feature matrix
            confidence_level: Confidence level for interval
            
        Returns:
            Tuple of (predicted_yield, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        # Get predictions from all trees (for Random Forest)
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_]).T[0]
            
            # Get mean and std
            mean_prediction = np.mean(tree_predictions)
            std_prediction = np.std(tree_predictions)
            
            # Calculate confidence interval (assuming normal distribution)
            from scipy import stats
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = mean_prediction - z_score * std_prediction
            upper = mean_prediction + z_score * std_prediction
            
            return (mean_prediction, max(0, lower), upper)
        else:
            # For Decision Tree, just return prediction
            prediction = self.predict(X)
            return (prediction[0], max(0, prediction[0] * 0.8), prediction[0] * 1.2)

