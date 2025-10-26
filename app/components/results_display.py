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


def render_fertilizer_results(recommendations: List[Tuple[str, float]], input_data: Dict):
    """
    Display fertilizer recommendations.
    
    Args:
        recommendations: List of (fertilizer_type, confidence) tuples
        input_data: User input parameters
    """
    st.markdown("---")
    st.markdown("### 🧪 Recommended Fertilizer")
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    # Display top recommendation prominently
    top_fertilizer, top_confidence = recommendations[0]
    st.success(f"🎯 **Top Recommendation: {top_fertilizer}** (Confidence: {top_confidence:.1%})")
    
    # Display all recommendations
    st.markdown("#### All Recommendations")
    cols = st.columns(min(3, len(recommendations)))
    colors = ['#3498db', '#2980b9', '#1f618d']
    
    for idx, (fertilizer, confidence) in enumerate(recommendations[:3]):
        with cols[idx]:
            st.markdown(f"""
                <div style="background-color: {colors[idx]}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">{fertilizer}</h3>
                    <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show confidence scores chart
    fig = go.Figure(data=[
        go.Bar(
            x=[fert for fert, _ in recommendations],
            y=[conf for _, conf in recommendations],
            marker_color='#3498db',
            text=[f"{conf:.1%}" for _, conf in recommendations],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Fertilizer Recommendation Confidence',
        xaxis_title='Fertilizer Type',
        yaxis_title='Confidence Score',
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show current soil nutrients
    st.markdown("### 📊 Current Soil Nutrients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nitrogen (N)", f"{input_data.get('Nitrogen', 0)} units")
    with col2:
        st.metric("Phosphorous (P)", f"{input_data.get('Phosphorous', 0)} units")
    with col3:
        st.metric("Potassium (K)", f"{input_data.get('Potassium', 0)} units")


def render_yield_results(predicted_production: float, input_data: Dict):
    """
    Display yield estimation results.
    
    Args:
        predicted_production: Predicted production value
        input_data: User input parameters
    """
    st.markdown("---")
    st.markdown("### 📊 Production Estimation Results")
    
    # Calculate metrics
    area = input_data.get('Area', 100)
    predicted_yield = predicted_production / area if area > 0 else 0
    
    # Display main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌾 Total Production", f"{predicted_production:.2f} tons")
    with col2:
        st.metric("📏 Yield per Hectare", f"{predicted_yield:.2f} tons/ha")
    with col3:
        st.metric("📍 Area", f"{area:.1f} ha")
    
    st.markdown("---")
    
    # Show visualization
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Production gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_production,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Production (tons)"},
            gauge = {
                'axis': {'range': [None, predicted_production * 1.5]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, predicted_production * 0.5], 'color': "lightgray"},
                    {'range': [predicted_production * 0.5, predicted_production], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_production * 0.9}}
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # Show input parameters summary
        st.markdown("#### 📊 Input Summary")
        st.write(f"**Crop:** {input_data.get('Crop', 'N/A')}")
        st.write(f"**Season:** {input_data.get('Season', 'N/A')}")
        st.write(f"**Year:** {input_data.get('Crop_Year', 'N/A')}")
        st.write(f"**State:** {input_data.get('State', 'N/A')}")
        st.write(f"**Rainfall:** {input_data.get('Annual_Rainfall', 0):.1f} mm")


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

