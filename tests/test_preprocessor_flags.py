import joblib
from pathlib import Path


MODEL_FOLDERS = {
    'crop': Path('models/crop_recommendation'),
    'fertilizer': Path('models/fertilizer_prediction'),
    'yield': Path('models/yield_estimation'),
}


def test_preprocessors_have_flags_and_encoders():
    """Ensure each saved preprocessor contains the has_categorical flag and encoders when expected."""
    for key, folder in MODEL_FOLDERS.items():
        pre_path = folder / 'preprocessor.pkl'
        assert pre_path.exists(), f"Preprocessor missing for {key}: {pre_path}"

        pp = joblib.load(pre_path)
        assert 'has_categorical' in pp, f"'has_categorical' not in preprocessor for {key}"
        assert isinstance(pp['has_categorical'], bool), f"has_categorical not bool for {key}"

        encoders = pp.get('label_encoders', {})
        assert isinstance(encoders, dict), f"label_encoders must be a dict for {key}"

        if pp['has_categorical']:
            # If there are categorical features, we expect at least one encoder
            assert len(encoders) > 0, f"Expected label_encoders for {key} but none found"
        else:
            # Numeric-only models should not have encoders
            assert len(encoders) == 0, f"Expected no encoders for numeric-only model {key}"

        # Basic sanity for feature names
        fn = pp.get('feature_names', None)
        assert fn is not None, f"feature_names missing for {key}"
        assert isinstance(fn, (list, tuple)), f"feature_names should be a list or tuple for {key}"
