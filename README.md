# 🌱 Smart Crop Recommendation System

An intelligent machine learning system that provides crop recommendations, fertilizer suggestions, and yield predictions based on soil and environmental parameters.

## 📋 Features

- **Crop Recommendation:** Suggests the best crops based on NPK values, pH, temperature, humidity, and rainfall
- **Fertilizer Prediction:** Recommends appropriate fertilizer types and quantities based on soil composition
- **Yield Estimation:** Predicts expected crop yield (kg/hectare) based on various parameters
- **Interactive UI:** User-friendly Streamlit interface with real-time predictions
- **Modular Design:** Easy to extend and update with advanced models

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   # Recommended: create a project-local virtual environment named `.venv`
   python -m venv .venv
   # Activate on Linux/macOS (zsh or bash)
   source .venv/bin/activate
   # On Windows (PowerShell)
   # .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download datasets from Kaggle and place them in `data/raw/` directory:
   - Crop Recommendation Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
   - Fertilizer Prediction Dataset: https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction
   - Yield Prediction Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

### Running the Application

```bash
streamlit run app/app.py
```

The application will open in your default browser at `http://localhost:8501`

## 🧰 Development environment (recommended)

We recommend running the app and tests inside the project virtual environment so runtime dependencies (for example, joblib and scikit-learn) are available to Streamlit and the diagnostics UI.

- Create & activate (Linux/macOS / zsh):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

- Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

- Run tests (from project root):
   ```bash
   .venv/bin/pytest -q
   ```

- Run the Streamlit app from the same environment (ensures diagnostics and model loading work):
   ```bash
   .venv/bin/streamlit run app/app.py
   ```

Note: If Streamlit is launched outside the project virtualenv, the app may display "Preprocessor diagnostics unavailable" or fail to load saved preprocessors because required packages (joblib, scikit-learn) are not on the system Python path. Using the project `.venv` avoids this issue.

## 📁 Project Structure

```
CropPrediction/
├── data/                    # Data storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Preprocessed data
│   └── external/           # External data cache
├── models/                 # Trained models
│   ├── crop_recommendation/
│   ├── fertilizer_prediction/
│   └── yield_estimation/
├── src/                    # Source code
│   ├── data/              # Data utilities
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── training/          # Training pipeline
│   └── utils/             # Helper utilities
├── app/                    # Streamlit application
│   ├── components/        # UI components
│   └── assets/           # Static assets
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── config.yaml           # Configuration file
```

## 🎯 Usage

1. **Input Parameters:** Enter your soil and environmental parameters (N, P, K, pH, temperature, humidity, rainfall)
2. **Get Recommendations:** Click "Get Recommendations" to receive:
   - Top 3 crop recommendations with confidence scores
   - Recommended fertilizer type and quantity
   - Expected yield estimation
3. **View Insights:** Explore visualizations showing feature importance and parameter comparisons

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model hyperparameters
- Data paths
- API settings
- Streamlit appearance

## 🤖 Models

- **Crop Recommendation:** Random Forest Classifier
- **Fertilizer Prediction:** Random Forest Classifier
- **Yield Estimation:** Random Forest Regressor

All models use Scikit-learn and can be easily swapped with more advanced models in the future.

## 📊 Dataset Information

- **Crop Recommendation:** 2200 samples, 22 crop types, 7 features
- **Fertilizer Prediction:** Soil and crop-based fertilizer recommendations
- **Yield Prediction:** Historical yield data with state/district/season information

## 🛠️ Development

### Training Models

Run the training pipeline:

```bash
python -m src.training.train
```

### Notebooks

Explore data and model development in Jupyter notebooks:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_model_training.ipynb`
- `notebooks/03_model_evaluation.ipynb`

## 🔮 Future Enhancements

- Real-time weather API integration
- Advanced ML models (XGBoost, Neural Networks)
- Disease detection capabilities
- Mobile app deployment
- Multi-language support

## 📝 License

This project is open source and available for educational purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on the repository.

---

Built with ❤️ using Python, Scikit-learn, and Streamlit

