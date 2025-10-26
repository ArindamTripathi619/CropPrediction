"""
Results display components for predictions.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict


def render_crop_recommendations(recommendations: List[Tuple[str, float]], input_data: Dict):
    """
    Display crop recommendations.
    
    Args:
        recommendations: List of (crop_name, confidence) tuples
        input_data: User input parameters
    """
    st.markdown("---")
    st.markdown("### 🌾 Recommended Crops")
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    # Display top 3 recommendations
    cols = st.columns(3)
    colors = ['#2ecc71', '#27ae60', '#229954']
    
    for idx, (crop, confidence) in enumerate(recommendations):
        with cols[idx]:
            st.markdown(f"""
                <div style="background-color: {colors[idx]}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">{crop}</h3>
                    <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show confidence scores chart
    fig = go.Figure(data=[
        go.Bar(
            x=[crop for crop, _ in recommendations],
            y=[conf for _, conf in recommendations],
            marker_color='#2ecc71',
            text=[f"{conf:.1%}" for _, conf in recommendations],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Crop Recommendation Confidence',
        xaxis_title='Crop',
        yaxis_title='Confidence Score',
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show input parameters
    st.markdown("### 📊 Your Input Parameters")
    show_parameters(input_data)


def render_fertilizer_results(fertilizer_recommendation: Dict):
    """
    Display fertilizer recommendations.
    
    Args:
        fertilizer_recommendation: Dictionary of fertilizer recommendations
    """
    st.markdown("---")
    st.markdown("### 🧪 Recommended Fertilizer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nitrogen (N)", f"{fertilizer_recommendation.get('N', 0)} kg")
    with col2:
        st.metric("Phosphorus (P)", f"{fertilizer_recommendation.get('P', 0)} kg")
    with col3:
        st.metric("Potassium (K)", f"{fertilizer_recommendation.get('K', 0)} kg")


def render_yield_results(yield_estimation: Dict):
    """
    Display yield estimation results.
    
    Args:
        yield_estimation: Dictionary of yield estimation data
    """
    st.markdown("---")
    st.markdown("### 📊 Estimated Yield")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Yield", f"{yield_estimation.get('predicted', 0):.1f} kg/ha")
        st.metric("Lower Bound", f"{yield_estimation.get('lower', 0):.1f} kg/ha")
        st.metric("Upper Bound", f"{yield_estimation.get('upper', 0):.1f} kg/ha")
    
    with col2:
        # Show confidence range
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Yield Estimate'],
            y=[yield_estimation.get('predicted', 0)],
            name='Predicted',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title='Yield Estimation',
            yaxis_title='Yield (kg/ha)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_parameters(params: Dict):
    """
    Display user input parameters.
    
    Args:
        params: Dictionary of parameters
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environmental Parameters:**")
        for key, value in list(params.items())[:4]:
            st.write(f"- {key}: {value}")
    
    with col2:
        st.markdown("**Soil Composition:**")
        for key, value in list(params.items())[4:]:
            st.write(f"- {key}: {value}")


def render_model_performance(metrics: Dict):
    """
    Display model performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metric_items = list(metrics.items())
    
    for i, col in enumerate([col1, col2, col3, col4]):
        if i < len(metric_items):
            key, value = metric_items[i]
            with col:
                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")

