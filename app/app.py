"""
Main Streamlit application for Smart Crop Recommendation System.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components using relative imports
from components.input_form import render_input_form, get_user_input
from components.results_display import render_crop_recommendations, render_fertilizer_results, render_yield_results
from src.models.crop_model import CropRecommendationModel
from src.models.fertilizer_model import FertilizerPredictionModel
from src.models.yield_model import YieldEstimationModel
from src.utils.config import get_config
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🌱 Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #2ecc71;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            border-radius: 0.5rem;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'crop_model' not in st.session_state:
    st.session_state.crop_model = None
if 'fertilizer_model' not in st.session_state:
    st.session_state.fertilizer_model = None
if 'yield_model' not in st.session_state:
    st.session_state.yield_model = None


@st.cache_data
def load_models():
    """Load trained models."""
    try:
        config = get_config()
        
        # Load crop model
        crop_path = Path(config['models']['crop_recommendation']['path'])
        if crop_path.exists():
            crop_model = CropRecommendationModel(
                **config['models']['crop_recommendation']['params']
            )
            crop_model.load_model(str(crop_path))
        else:
            st.error("Crop model not found. Please train the models first.")
            return None, None, None
        
        # Load fertilizer model
        fertilizer_path = Path(config['models']['fertilizer_prediction']['path'])
        if fertilizer_path.exists():
            fertilizer_model = FertilizerPredictionModel(
                **config['models']['fertilizer_prediction']['params']
            )
            fertilizer_model.load_model(str(fertilizer_path))
        else:
            st.warning("Fertilizer model not found.")
            fertilizer_model = None
        
        # Load yield model
        yield_path = Path(config['models']['yield_estimation']['path'])
        if yield_path.exists():
            yield_model = YieldEstimationModel(
                **config['models']['yield_estimation']['params']
            )
            yield_model.load_model(str(yield_path))
        else:
            st.warning("Yield model not found.")
            yield_model = None
        
        return crop_model, fertilizer_model, yield_model
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


def main():
    """Main application function."""
    
    # Sidebar
    with st.sidebar:
        st.title("🌱 Smart Crop Recommender")
        st.markdown("---")
        
        st.markdown("### Navigation")
        page = st.selectbox(
            "Select a page",
            ["🏠 Home", "🌾 Crop Recommendation", "🧪 Fertilizer Prediction", "📊 Yield Estimation", "📈 Model Performance"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Smart Crop Recommendation System**
        
        Get intelligent recommendations for:
        - Best crops to grow
        - Required fertilizers
        - Expected yield
        
        Using advanced ML algorithms!
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "🌾 Crop Recommendation":
        show_crop_recommendation()
    elif page == "🧪 Fertilizer Prediction":
        show_fertilizer_prediction()
    elif page == "📊 Yield Estimation":
        show_yield_estimation()
    elif page == "📈 Model Performance":
        show_model_performance()


def show_home():
    """Show home page."""
    st.markdown("<h1 class='main-header'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Smart Crop Recommendation System!
    
    This intelligent system helps farmers and agriculture enthusiasts make data-driven decisions by providing:
    
    - **🌾 Crop Recommendations**: Based on your soil composition and environmental factors
    - **🧪 Fertilizer Suggestions**: Optimal fertilizer types and quantities
    - **📊 Yield Predictions**: Expected crop yields based on multiple parameters
    
    ### How It Works
    
    1. Enter your soil and environmental parameters (NPK values, pH, temperature, etc.)
    2. Our trained ML models analyze the data
    3. Get personalized recommendations for your farm
    
    ### Getting Started
    
    Navigate to the different sections from the sidebar to explore each feature:
    
    - **Crop Recommendation**: Discover which crops are best for your conditions
    - **Fertilizer Prediction**: Learn what fertilizers you need
    - **Yield Estimation**: Estimate potential crop yields
    
    ---
    
    ### Quick Demo
    
    Try these sample scenarios by navigating to the recommendation pages:
    
    **Scenario 1: Rich soil, moderate climate**
    - N: 90, P: 40, K: 40
    - pH: 6.5, Temperature: 25°C
    - Humidity: 80%, Rainfall: 200mm
    
    **Scenario 2: Nutrient-deficient soil**
    - N: 20, P: 10, K: 15
    - pH: 7.0, Temperature: 28°C
    - Humidity: 70%, Rainfall: 150mm
    """)


def show_crop_recommendation():
    """Show crop recommendation page."""
    st.title("🌾 Crop Recommendation")
    st.markdown("Enter your soil and environmental parameters to get crop recommendations.")
    
    # Render input form
    input_data = render_input_form()
    
    if st.button("Get Crop Recommendations", type="primary"):
        if input_data is not None:
            with st.spinner("Analyzing your data and generating recommendations..."):
                try:
                    # Load models if needed
                    if not st.session_state.models_loaded:
                        crop_model, _, _ = load_models()
                        if crop_model:
                            st.session_state.crop_model = crop_model
                            st.session_state.models_loaded = True
                        else:
                            st.error("Could not load models. Please train models first.")
                            return
                    
                    # Prepare input - create DataFrame from user input
                    input_df = pd.DataFrame([input_data])
                    
                    # Apply feature engineering (creates all features including booleans)
                    from src.features.feature_engineering import engineer_features
                    input_df_engineered = engineer_features(input_df)
                    
                    # Load and apply preprocessor for scaling
                    import joblib
                    preprocessor_path = Path("models/crop_recommendation/preprocessor.pkl")
                    if preprocessor_path.exists():
                        preprocessor_data = joblib.load(preprocessor_path)
                        scaler = preprocessor_data['scaler']
                        feature_names = preprocessor_data['feature_names']
                        
                        # Scale only the features that were scaled during training (non-boolean features)
                        # Boolean features are kept as-is
                        input_df_processed = input_df_engineered.copy()
                        input_df_processed[feature_names] = scaler.transform(input_df_engineered[feature_names])
                    else:
                        input_df_processed = input_df_engineered
                    
                    # Ensure features are in the same order as training
                    expected_features = st.session_state.crop_model.model.feature_names_in_
                    input_df_processed = input_df_processed[expected_features]
                    
                    # Get recommendations
                    recommendations = st.session_state.crop_model.get_top_predictions(input_df_processed, top_k=3)
                    
                    # Display results
                    render_crop_recommendations(recommendations, input_data)
                
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")


def show_fertilizer_prediction():
    """Show fertilizer prediction page."""
    st.title("🧪 Fertilizer Prediction")
    st.markdown("Predict optimal fertilizer requirements for your crops.")
    
    # Render input form
    input_data = render_input_form()
    
    if st.button("Get Fertilizer Recommendations", type="primary"):
        if input_data is not None:
            with st.spinner("Analyzing soil composition..."):
                st.info("Fertilizer prediction will be available after training the fertilizer model.")


def show_yield_estimation():
    """Show yield estimation page."""
    st.title("📊 Yield Estimation")
    st.markdown("Estimate potential crop yields based on your parameters.")
    
    # Render input form
    input_data = render_input_form()
    
    if st.button("Estimate Yield", type="primary"):
        if input_data is not None:
            with st.spinner("Calculating yield estimates..."):
                st.info("Yield estimation will be available after training the yield model.")


def show_model_performance():
    """Show model performance page."""
    st.title("📈 Model Performance")
    st.markdown("View performance metrics for trained models.")
    
    st.info("Model performance metrics will be available after training models.")
    
    # Placeholder for model performance visualization
    st.markdown("### Performance Metrics")
    st.markdown("""
    **Target Metrics:**
    - Crop Recommendation Accuracy: >85%
    - Fertilizer Prediction Accuracy: >80%
    - Yield Estimation R² Score: >0.75
    """)


if __name__ == "__main__":
    main()

