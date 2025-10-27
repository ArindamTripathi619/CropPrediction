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
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load centralized CSS file (assets/styles.css) for consistent theming
try:
    css_path = Path(__file__).parent / 'assets' / 'styles.css'
    if css_path.exists():
        css_text = css_path.read_text()
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    else:
        # Fallback minimal inline styles if file missing
        st.markdown(
            """
            <style>
                .main-header { font-size: 2.5rem; color: #2ecc71; text-align: center; margin-bottom: 1.5rem; }
                .stButton>button { background-color: #2ecc71; color: white; border-radius: 0.5rem; width: 100%; }
            </style>
            """,
            unsafe_allow_html=True,
        )
except Exception:
    # Don't break app if CSS injection fails
    pass

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
    # Render header
    try:
        st.markdown(
            """
            <div class="app-header card">
                <div class="branding">
                    <div class="logo">🌱</div>
                    <div>
                        <div class="title">Smart Crop Recommendation</div>
                        <div class="tagline">Data-driven crop, fertilizer and yield guidance</div>
                    </div>
                </div>
                <div class="actions">
                    <a href="https://github.com/ArindamTripathi619/CropPrediction" target="_blank" style="color:var(--primary); text-decoration:none; margin-right:12px;">Source</a>
                    <a href="README.md" target="_blank" style="color:var(--muted-2); text-decoration:none;">Docs</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass
    # On startup, check for model/preprocessor artifacts and surface issues in the sidebar
    artifact_check = check_model_artifacts()
    critical = artifact_check.get('critical', [])
    info = artifact_check.get('info', [])

    with st.sidebar:
        if critical:
            st.error("Some model artifacts are missing or incomplete. See details below and follow the DEV_SETUP.md to generate them.")
            for item in critical:
                st.write(f"- {item}")
            st.markdown("\n**How to fix:** run the training pipeline to generate model and preprocessor artifacts, or copy trained artifacts into the `models/` folder.")
        if info:
            st.info("Additional informational checks: these are generally non-critical but may affect categorical inputs. See details below.")
            for item in info:
                st.write(f"- {item}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>🌱 Smart Crop Recommender</div>", unsafe_allow_html=True)
        st.markdown("---")
        # Compact navigation with icons — keep existing pages for routing
        st.markdown("### Navigation")
        # Provide a non-empty label for accessibility; keep it visually hidden because
        # we already render a visible "Navigation" heading above.
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🌾 Crop Recommendation", "🧪 Fertilizer Prediction", "📊 Yield Estimation", "📈 Model Performance"],
            index=0,
            label_visibility='collapsed',
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

    # Footer (global)
    try:
        st.markdown(
            """
            <div class='app-footer'>
                Built with ❤️ · <a href='https://github.com/ArindamTripathi619/CropPrediction' target='_blank' style='color:var(--muted-2)'>GitHub</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass


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
    
    # Import fertilizer input form
    from components.input_form import render_fertilizer_input_form
    
    # Render input form
    input_data = render_fertilizer_input_form()
    
    if st.button("Get Fertilizer Recommendations", type="primary"):
        if input_data is not None:
            with st.spinner("Analyzing soil composition..."):
                try:
                    # Load models if needed
                    if not st.session_state.models_loaded:
                        _, fertilizer_model, _ = load_models()
                        if fertilizer_model:
                            st.session_state.fertilizer_model = fertilizer_model
                            st.session_state.models_loaded = True
                        else:
                            st.error("Could not load fertilizer model.")
                            return
                    
                    # Check if fertilizer model is loaded
                    if st.session_state.fertilizer_model is None:
                        _, fertilizer_model, _ = load_models()
                        st.session_state.fertilizer_model = fertilizer_model
                    
                    # Prepare input
                    input_df = pd.DataFrame([input_data])

                    # Ensure preprocessor exists before attempting to predict (categorical features must be encoded)
                    import joblib
                    preprocessor_path = Path("models/fertilizer_prediction/preprocessor.pkl")
                    if not preprocessor_path.exists():
                        st.error("Preprocessor for fertilizer model not found. Please train the fertilizer model and save the preprocessor to 'models/fertilizer_prediction/preprocessor.pkl'.")
                        return

                    # Load preprocessor to get label encoders
                    preprocessor_data = joblib.load(preprocessor_path)
                    label_encoders = preprocessor_data.get('label_encoders', {})

                    # Encode categorical features; if encoder missing, show helpful error
                    if 'Soil Type' in input_df.columns:
                        if 'Soil Type' in label_encoders:
                            input_df['Soil Type'] = label_encoders['Soil Type'].transform(input_df['Soil Type'])
                        else:
                            st.error("Label encoder for 'Soil Type' not found in preprocessor. Please ensure the preprocessor contains fitted label encoders.")
                            return
                    if 'Crop Type' in input_df.columns:
                        if 'Crop Type' in label_encoders:
                            input_df['Crop Type'] = label_encoders['Crop Type'].transform(input_df['Crop Type'])
                        else:
                            st.error("Label encoder for 'Crop Type' not found in preprocessor. Please ensure the preprocessor contains fitted label encoders.")
                            return

                    # Get prediction
                    prediction = st.session_state.fertilizer_model.predict(input_df)
                    
                    # Get prediction probabilities if available
                    if hasattr(st.session_state.fertilizer_model, 'get_top_predictions'):
                        recommendations = st.session_state.fertilizer_model.get_top_predictions(input_df, top_k=3)
                    else:
                        recommendations = [(prediction[0], 1.0)]
                    
                    # Display results
                    render_fertilizer_results(recommendations, input_data)
                
                except Exception as e:
                    st.error(f"Error getting fertilizer recommendations: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def show_yield_estimation():
    """Show yield estimation page."""
    st.title("📊 Yield Estimation")
    st.markdown("Estimate potential crop yields based on your parameters.")
    
    # Import yield input form
    from components.input_form import render_yield_input_form
    
    # Render input form
    input_data = render_yield_input_form()
    
    if st.button("Estimate Yield", type="primary"):
        if input_data is not None:
            with st.spinner("Calculating yield estimates..."):
                try:
                    # Load models if needed
                    if not st.session_state.models_loaded:
                        _, _, yield_model = load_models()
                        if yield_model:
                            st.session_state.yield_model = yield_model
                            st.session_state.models_loaded = True
                        else:
                            st.error("Could not load yield model.")
                            return
                    
                    # Check if yield model is loaded
                    if st.session_state.yield_model is None:
                        _, _, yield_model = load_models()
                        st.session_state.yield_model = yield_model
                    
                    # Prepare input
                    input_df = pd.DataFrame([input_data])

                    # Ensure preprocessor exists before attempting to predict (categorical features must be encoded)
                    import joblib
                    preprocessor_path = Path("models/yield_estimation/preprocessor.pkl")
                    if not preprocessor_path.exists():
                        st.error("Preprocessor for yield model not found. Please train the yield model and save the preprocessor to 'models/yield_estimation/preprocessor.pkl'.")
                        return

                    # Load preprocessor to get label encoders
                    preprocessor_data = joblib.load(preprocessor_path)
                    label_encoders = preprocessor_data.get('label_encoders', {})

                    # Encode categorical features; if encoder missing, show helpful error
                    for col in ['State', 'District', 'Season', 'Crop']:
                        if col in input_df.columns:
                            if col in label_encoders:
                                try:
                                    input_df[col] = label_encoders[col].transform(input_df[col])
                                except Exception:
                                    # If value not seen during training, use most common class (0) as fallback
                                    input_df[col] = 0
                            else:
                                st.error(f"Label encoder for '{col}' not found in preprocessor. Please ensure the preprocessor contains fitted label encoders.")
                                return

                    # Get prediction
                    prediction = st.session_state.yield_model.predict(input_df)
                    
                    # Display results
                    render_yield_results(prediction[0], input_data)
                
                except Exception as e:
                    st.error(f"Error estimating yield: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def show_model_performance():
    """Show model performance page."""
    st.title("📈 Model Performance")
    st.markdown("View performance metrics for trained models.")

    st.info("Model performance metrics will be available after training models.")

    # Show saved preprocessor flags for transparency
    st.markdown("### Preprocessor flags & quick diagnostics")
    try:
        import joblib
        from pathlib import Path

        model_sets = [
            ('Crop Recommendation', Path('models/crop_recommendation')),
            ('Fertilizer Prediction', Path('models/fertilizer_prediction')),
            ('Yield Estimation', Path('models/yield_estimation')),
        ]

        for name, folder in model_sets:
            with st.expander(name):
                pre = folder / 'preprocessor.pkl'
                if not pre.exists():
                    st.warning(f"Preprocessor not found for {name} ({pre}).")
                    continue

                try:
                    pp = joblib.load(pre)
                except Exception as e:
                    st.error(f"Failed to load preprocessor for {name}: {e}")
                    continue

                has_cat = pp.get('has_categorical', None)
                encoders = pp.get('label_encoders', {})
                fn = pp.get('feature_names', [])

                st.write(f"has_categorical: {has_cat}")
                st.write(f"num_label_encoders: {len(encoders) if isinstance(encoders, dict) else 'N/A'}")
                st.write(f"num_feature_names: {len(fn) if fn else 0}")
                if isinstance(encoders, dict) and len(encoders) > 0:
                    st.markdown("**Encoders:**")
                    for k in encoders.keys():
                        st.write(f"- {k}")

    except Exception:
        st.info("Preprocessor diagnostics unavailable (missing joblib or other error).")

    # Placeholder for target metrics
    st.markdown("### Performance Metrics")
    st.markdown("""
    **Target Metrics:**
    - Crop Recommendation Accuracy: >85%
    - Fertilizer Prediction Accuracy: >80%
    - Yield Estimation R² Score: >0.75
    """)




# Utility: check whether model and preprocessor artifacts exist and look reasonable
def check_model_artifacts():
    """Return a dict with 'critical' and 'info' lists describing issues found.

    - critical: problems that will almost certainly break runtime predictions (missing files,
      corrupt preprocessor, missing feature_names, etc.)
    - info: informational notes that may be expected (e.g. empty label_encoders when a model
      has no categorical inputs). These are presented as info in the UI so users aren't
      unnecessarily alarmed.
    """
    result = {'critical': [], 'info': []}
    try:
        import joblib
    except Exception:
        # If joblib isn't available, nothing to check at runtime
        return result

    model_sets = [
        ('Crop Recommendation', Path('models/crop_recommendation')),
        ('Fertilizer Prediction', Path('models/fertilizer_prediction')),
        ('Yield Estimation', Path('models/yield_estimation')),
    ]

    for name, folder in model_sets:
        if not folder.exists():
            result['critical'].append(f"{name}: folder missing ({folder})")
            continue

        pre = folder / 'preprocessor.pkl'
        if not pre.exists():
            result['critical'].append(f"{name}: preprocessor.pkl not found in {folder}")
            continue

        # Try loading and probing the preprocessor
        try:
            pp = joblib.load(pre)
            if not isinstance(pp, dict):
                result['critical'].append(f"{name}: preprocessor loaded but structure unexpected (not a dict)")
                continue

            # Check feature_names exist (critical)
            fn = pp.get('feature_names')
            if not fn:
                result['critical'].append(f"{name}: preprocessor.feature_names missing or empty")

            # Check label_encoders: empty encoders are informational (common when model has
            # purely numeric inputs). For the crop recommender we commonly expect numeric-only
            # features; skip the informational message for that case to avoid noise.
            # Prefer using an explicit flag saved by the preprocessor when available
            has_cat = pp.get('has_categorical', None)
            if has_cat is False:
                # Saved preprocessor says there are no categorical features — nothing to warn about
                pass
            elif has_cat is True:
                # Preprocessor indicates categorical features were present; ensure encoders exist
                le = pp.get('label_encoders', {})
                if isinstance(le, dict) and len(le) == 0:
                    result['info'].append(f"{name}: preprocessor.has_categorical=True but label_encoders is empty — categorical fields won't be encoded.")
            else:
                # Backwards compatibility: fall back to previous heuristic
                le = pp.get('label_encoders', {})
                if isinstance(le, dict) and len(le) == 0:
                    if name != 'Crop Recommendation':
                        result['info'].append(f"{name}: preprocessor.label_encoders is empty — categorical fields won't be encoded. This is OK if the model has no categorical inputs.")

        except Exception as e:
            result['critical'].append(f"{name}: failed to load preprocessor ({e})")

    return result


if __name__ == "__main__":
    main()

