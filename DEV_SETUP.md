Developer setup (Arch Linux / Linux)

This file documents how to set up the development environment, run tests, and (re)generate model artifacts.

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

2) Run tests

```bash
# from project root
.venv/bin/python -m pytest -q
```

3) Training and producing model artifacts

The Streamlit app expects the trained models and preprocessors at the following paths:

- `models/crop_recommendation/rf_model.pkl`
- `models/crop_recommendation/preprocessor.pkl`
- `models/fertilizer_prediction/rf_model.pkl`
- `models/fertilizer_prediction/preprocessor.pkl`
- `models/yield_estimation/rf_model.pkl`
- `models/yield_estimation/preprocessor.pkl`

If you have the datasets locally, run the training pipeline (check `src/training/`) which will produce model and preprocessor pickles.

Typical pattern in training code (what the app expects):

- Train models on numeric features and categorical features encoded using `sklearn.preprocessing.LabelEncoder`.
- Save a `preprocessor.pkl` using `joblib.dump(...)` that contains at least these keys:
  - `scaler`: scaler object used for numeric scaling (optional)
  - `label_encoders`: dict mapping column name -> fitted LabelEncoder
  - `imputer`: any imputer used (optional)
  - `feature_names`: ordered list of features used for model training

Example to save preprocessor at the end of training:

```python
from src.data.preprocessor import DataPreprocessor
import joblib

preprocessor = {
    'scaler': data_preprocessor.scaler,
    'label_encoders': data_preprocessor.label_encoder,
    'imputer': data_preprocessor.imputer,
    'feature_names': data_preprocessor.feature_names
}
joblib.dump(preprocessor, 'models/fertilizer_prediction/preprocessor.pkl')
```

4) Dealing with categorical errors in the Streamlit app

If you see errors like "could not convert string to float: 'Loamy'" it means the model received raw strings. Fixes:

- Ensure the corresponding `preprocessor.pkl` exists and contains `label_encoders` with LabelEncoders for categorical fields.
- If you have trained artifacts, copy them into `models/...` as shown above.
- As a short-term workaround, the local environment may include best-effort encoders added by the developer, but the reliable fix is retraining and saving the correct preprocessor.

5) Normalize line endings (prevent huge diffs when copying between Windows and Linux)

Add `.gitattributes` with:

```
* text=auto
*.py text eol=lf
*.md text eol=lf
```

Run these commands to normalize (optional):

```bash
# create .gitattributes then
git add .gitattributes
# normalize line endings
git add --renormalize .
```

6) Notes

- Do not commit large model binary files unless you want them in the repo; consider storing them in releases or an artifacts storage.
- For CI, ensure the runner has Python and dependencies installed; you can create a smaller `requirements-test.txt` with only test deps if desired.
