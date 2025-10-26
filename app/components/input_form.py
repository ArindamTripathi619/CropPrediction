"""
Input form components for user data entry.
"""

import streamlit as st
from typing import Dict, Optional


def render_input_form() -> Optional[Dict]:
    """
    Render input form for user parameters.
    
    Returns:
        Dictionary of input parameters or None
    """
    st.markdown("### Enter Your Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Environmental Parameters")
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0, 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 200.0, 10.0)
        pH = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
    
    with col2:
        st.markdown("#### 🧪 Soil Composition")
        nitrogen = st.slider("Nitrogen (N)", 0, 150, 50, 1)
        phosphorus = st.slider("Phosphorus (P)", 0, 150, 40, 1)
        potassium = st.slider("Potassium (K)", 0, 150, 40, 1)
    
    # Additional parameters
    with st.expander("📊 Additional Information"):
        st.markdown("Optional parameters for more accurate predictions")
        # Can add more parameters here if needed
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'ph': pH,  # Use lowercase 'ph' to match training data
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium
    }


def get_user_input() -> Dict:
    """
    Get user input parameters.
    
    Returns:
        Dictionary of input parameters
    """
    return render_input_form()

