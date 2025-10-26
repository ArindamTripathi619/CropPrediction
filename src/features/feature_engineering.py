"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from typing import List


def create_npk_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create NPK ratio features.
    
    Args:
        df: Input DataFrame with N, P, K columns
        
    Returns:
        DataFrame: DataFrame with additional ratio features
    """
    df_engineered = df.copy()
    
    if all(col in df.columns for col in ['N', 'P', 'K']):
        # NPK ratios
        df_engineered['NP_ratio'] = df['N'] / (df['P'] + 1e-8)
        df_engineered['NK_ratio'] = df['N'] / (df['K'] + 1e-8)
        df_engineered['PK_ratio'] = df['P'] / (df['K'] + 1e-8)
        
        # Total NPK
        df_engineered['total_npk'] = df['N'] + df['P'] + df['K']
        
        # Balanced NPK indicator
        df_engineered['npk_balance'] = abs(df['N'] - df['P']) + abs(df['N'] - df['K']) + abs(df['P'] - df['K'])
    
    return df_engineered


def create_temperature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temperature range features.
    
    Args:
        df: Input DataFrame with temperature column
        
    Returns:
        DataFrame: DataFrame with additional range features
    """
    df_engineered = df.copy()
    
    if 'temperature' in df.columns:
        # Temperature categories
        df_engineered['temp_low'] = df['temperature'] < 20
        df_engineered['temp_medium'] = (df['temperature'] >= 20) & (df['temperature'] <= 30)
        df_engineered['temp_high'] = df['temperature'] > 30
        
        # Temperature deviation from optimal (assuming 25°C is optimal)
        df_engineered['temp_deviation'] = abs(df['temperature'] - 25)
    
    return df_engineered


def create_ph_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pH range features.
    
    Args:
        df: Input DataFrame with pH column
        
    Returns:
        DataFrame: DataFrame with additional pH features
    """
    df_engineered = df.copy()
    
    if 'pH' in df.columns:
        # pH categories (acidic, neutral, alkaline)
        df_engineered['ph_acidic'] = df['pH'] < 6.5
        df_engineered['ph_neutral'] = (df['pH'] >= 6.5) & (df['pH'] <= 7.5)
        df_engineered['ph_alkaline'] = df['pH'] > 7.5
        
        # pH deviation from neutral
        df_engineered['ph_deviation'] = abs(df['pH'] - 7.0)
    
    return df_engineered


def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rainfall-based features.
    
    Args:
        df: Input DataFrame with rainfall column
        
    Returns:
        DataFrame: DataFrame with additional rainfall features
    """
    df_engineered = df.copy()
    
    if 'rainfall' in df.columns:
        # Rainfall categories (low, medium, high)
        df_engineered['rainfall_low'] = df['rainfall'] < 200
        df_engineered['rainfall_medium'] = (df['rainfall'] >= 200) & (df['rainfall'] <= 1000)
        df_engineered['rainfall_high'] = df['rainfall'] > 1000
    
    return df_engineered


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with interaction features
    """
    df_engineered = df.copy()
    
    # Temperature × Humidity interaction
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df_engineered['temp_humidity'] = df['temperature'] * df['humidity']
    
    # pH × NPK interactions
    if 'pH' in df.columns and 'N' in df.columns:
        df_engineered['ph_n'] = df['pH'] * df['N']
    if 'pH' in df.columns and 'P' in df.columns:
        df_engineered['ph_p'] = df['pH'] * df['P']
    
    # Rainfall × Temperature interaction
    if 'rainfall' in df.columns and 'temperature' in df.columns:
        df_engineered['rainfall_temp'] = df['rainfall'] * df['temperature']
    
    return df_engineered


def engineer_features(df: pd.DataFrame, 
                     include_ratios: bool = True,
                     include_ranges: bool = True,
                     include_interactions: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        include_ratios: Whether to include ratio features
        include_ranges: Whether to include range features
        include_interactions: Whether to include interaction features
        
    Returns:
        DataFrame: DataFrame with all engineered features
    """
    df_engineered = df.copy()
    
    if include_ratios:
        df_engineered = create_npk_ratios(df_engineered)
    
    if include_ranges:
        df_engineered = create_temperature_ranges(df_engineered)
        df_engineered = create_ph_ranges(df_engineered)
        df_engineered = create_rainfall_features(df_engineered)
    
    if include_interactions:
        df_engineered = create_interaction_features(df_engineered)
    
    return df_engineered


def get_feature_importance(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        df: DataFrame with feature columns
        model: Trained model with feature_importances_ attribute
        
    Returns:
        DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': df.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    else:
        print("Model does not support feature importance.")
        return None


def select_top_features(df: pd.DataFrame, model, top_n: int = 10) -> List[str]:
    """
    Select top N features based on importance.
    
    Args:
        df: DataFrame with all features
        model: Trained model
        top_n: Number of top features to select
        
    Returns:
        List of top feature names
    """
    importance_df = get_feature_importance(df, model)
    if importance_df is not None:
        top_features = importance_df.head(top_n)['feature'].tolist()
        return top_features
    else:
        return df.columns.tolist()[:top_n]

