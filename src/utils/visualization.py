"""
Visualization utilities for the crop recommendation system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np


def set_style():
    """Set consistent plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        fig: Matplotlib figure object
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        fig: Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(models, metrics):
    """
    Plot comparison of model metrics.
    
    Args:
        models: List of model names
        metrics: Dictionary of metric values
        
    Returns:
        fig: Plotly figure object
    """
    fig = go.Figure()
    
    for metric_name, values in metrics.items():
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            name=metric_name,
            text=values,
            texttemplate='%{text:.3f}',
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig


def create_crop_recommendation_chart(recommendations):
    """
    Create a visualization of crop recommendations.
    
    Args:
        recommendations: List of tuples (crop_name, confidence_score)
        
    Returns:
        fig: Plotly figure object
    """
    crops = [r[0] for r in recommendations]
    scores = [r[1] for r in recommendations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=crops,
            y=scores,
            marker_color='#2ecc71',
            text=scores,
            texttemplate='%{text:.2%}',
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Crop Recommendations',
        xaxis_title='Crop',
        yaxis_title='Confidence Score',
        height=400
    )
    
    return fig


def plot_parameter_comparison(user_input, optimal_ranges):
    """
    Compare user input with optimal parameter ranges.
    
    Args:
        user_input: Dictionary of user input parameters
        optimal_ranges: Dictionary of optimal parameter ranges
        
    Returns:
        fig: Plotly figure object
    """
    params = list(user_input.keys())
    user_values = [user_input[p] for p in params]
    optimal_min = [optimal_ranges[p][0] for p in params]
    optimal_max = [optimal_ranges[p][1] for p in params]
    
    fig = go.Figure()
    
    # Optimal range
    fig.add_trace(go.Scatter(
        x=params,
        y=optimal_max,
        mode='lines+markers',
        name='Optimal Max',
        line=dict(color='green', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(46, 204, 113, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=params,
        y=optimal_min,
        mode='lines+markers',
        name='Optimal Min',
        line=dict(color='green', dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)'
    ))
    
    # User input
    fig.add_trace(go.Scatter(
        x=params,
        y=user_values,
        mode='markers',
        name='Your Input',
        marker=dict(size=10, color='red', symbol='circle')
    ))
    
    fig.update_layout(
        title='Parameter Comparison',
        xaxis_title='Parameter',
        yaxis_title='Value',
        height=500,
        hovermode='x unified'
    )
    
    return fig

