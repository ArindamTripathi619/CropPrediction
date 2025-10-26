# 🚀 Quick Start Guide

## Step 1: Set Up Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Get the Datasets

You need to download three datasets from Kaggle:

### Option A: Automatic Download (with Kaggle API)
1. Install kaggle: `pip install kaggle`
2. Set up your credentials at `~/.kaggle/kaggle.json`
3. Run: `python scripts/download_datasets.py`

### Option B: Manual Download
Download these datasets manually and place them in `data/raw/`:

1. **Crop Recommendation Dataset**
   - URL: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
   - Download and rename to `crop_data.csv`

2. **Fertilizer Prediction Dataset**  
   - URL: https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction
   - Download and rename to `fertilizer_data.csv`

3. **Yield Prediction Dataset**
   - URL: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
   - Download and rename to `yield_data.csv`

## Step 3: Train the Models

Run the training pipeline:

```bash
python -m src.training.train
```

This will:
- Load and preprocess the data
- Train three models (crop, fertilizer, yield)
- Evaluate model performance
- Save trained models to `models/` directory

## Step 4: Run the Application

Start the Streamlit app:

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## Step 5: Use the Application

1. Navigate to "🌾 Crop Recommendation" page
2. Enter your soil and environmental parameters
3. Click "Get Crop Recommendations"
4. View your personalized recommendations!

## Project Structure

```
CropPrediction/
├── data/              # Datasets go here
│   └── raw/          # Place downloaded CSVs here
├── models/           # Trained models will be saved here
├── src/              # Source code
│   ├── data/        # Data loading & preprocessing
│   ├── models/      # ML model implementations
│   ├── training/    # Training pipeline
│   └── utils/       # Utilities
├── app/              # Streamlit application
└── notebooks/        # Jupyter notebooks for exploration
```

## Troubleshooting

**Problem:** Models not found error
- **Solution:** Run `python -m src.training.train` first

**Problem:** Import errors
- **Solution:** Make sure you're in the project root directory and have activated your virtual environment

**Problem:** Dataset not found
- **Solution:** Download datasets from Kaggle and place them in `data/raw/` with the correct names

## Next Steps

- Explore the notebooks in `notebooks/` directory
- Customize models in `config.yaml`
- Add your own features in `src/features/`
- Extend the UI in `app/`

Happy farming! 🌱

