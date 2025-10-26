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


def render_fertilizer_input_form() -> Optional[Dict]:
    """
    Render input form for fertilizer prediction.
    
    Returns:
        Dictionary of input parameters or None
    """
    st.markdown("### Enter Your Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Environmental Parameters")
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0, 1.0)
        moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0, 1.0)
        
        st.markdown("#### 🌱 Crop & Soil")
        soil_type = st.selectbox("Soil Type", ['Loamy', 'Clayey', 'Black', 'Sandy'])
        crop_type = st.selectbox("Crop Type", 
                                 ['Oil seeds', 'Maize', 'Paddy', 'Wheat', 'Cotton', 
                                  'Millets', 'Tobacco', 'Pulses', 'Sugarcane', 'Barley'])
    
    with col2:
        st.markdown("#### 🧪 Current Soil Nutrients (NPK)")
        nitrogen = st.slider("Nitrogen (N)", 0, 150, 50, 1)
        potassium = st.slider("Potassium (K)", 0, 150, 40, 1)
        phosphorous = st.slider("Phosphorous (P)", 0, 150, 40, 1)
    
    return {
        'Temparature': temperature,  # Note: keeping the typo to match training data
        'Humidity': humidity,
        'Moisture': moisture,
        'Soil Type': soil_type,
        'Crop Type': crop_type,
        'Nitrogen': nitrogen,
        'Potassium': potassium,
        'Phosphorous': phosphorous
    }


def render_yield_input_form() -> Optional[Dict]:
    """
    Render input form for yield estimation.
    
    Returns:
        Dictionary of input parameters or None
    """
    st.markdown("### Enter Your Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Location & Time")
        state = st.selectbox("State", 
                            ['Tamil Nadu', 'Gujarat', 'Maharashtra', 'Karnataka', 
                             'Andhra Pradesh', 'Uttar Pradesh', 'Punjab', 'Haryana',
                             'Rajasthan', 'Madhya Pradesh'])
        district = st.text_input("District", "District A")
        crop_year = st.slider("Crop Year", 2010, 2025, 2020, 1)
        season = st.selectbox("Season", ['Kharif', 'Rabi', 'Zaid', 'Whole Year'])
    
    with col2:
        st.markdown("#### 🌾 Crop Details")
        crop = st.selectbox("Crop", 
                           ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 
                            'Potato', 'Soyabean', 'Groundnut', 'Bajra', 'Jowar'])
        area = st.number_input("Area (hectares)", min_value=0.0, value=100.0, step=10.0)
        rainfall = st.slider("Annual Rainfall (mm)", 0.0, 3000.0, 1000.0, 10.0)
        yield_val = st.number_input("Expected Yield (tons/hectare)", min_value=0.0, value=2.0, step=0.1, 
                                    help="This is typically the target variable, but included for production calculation")
    
    return {
        'State': state,
        'District': district,
        'Crop_Year': crop_year,
        'Season': season,
        'Crop': crop,
        'Area': area,
        'Annual_Rainfall': rainfall,
        'Yield': yield_val
    }

