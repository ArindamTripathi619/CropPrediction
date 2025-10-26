# Test Suite Documentation

## Overview
Comprehensive test suite for the Smart Crop Recommendation System covering unit tests, integration tests, and model accuracy tests.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and shared utilities
├── test_data_preprocessing.py     # Data loading and preprocessing tests
├── test_feature_engineering.py    # Feature engineering tests
├── test_models.py                 # Basic model tests
├── test_model_accuracy.py         # Model accuracy and performance tests
└── README.md                      # This file
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
Test individual components in isolation:
- Data preprocessing functions
- Feature engineering functions
- Model initialization and basic operations

### Model Tests (`@pytest.mark.model`)
Test model training, predictions, and accuracy:
- Crop recommendation model
- Fertilizer prediction model
- Yield estimation model
- Cross-validation performance
- Model consistency

### Slow Tests (`@pytest.mark.slow`)
Tests that take longer to run:
- Training on full datasets
- Extensive cross-validation
- Performance benchmarking

## Running Tests

### Install Test Dependencies
```bash
pip install pytest pytest-cov
```

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only model tests
pytest -m model

# Run excluding slow tests
pytest -m "not slow"
```

### Run Specific Test Files
```bash
# Run preprocessing tests
pytest tests/test_data_preprocessing.py

# Run model accuracy tests
pytest tests/test_model_accuracy.py

# Run feature engineering tests
pytest tests/test_feature_engineering.py
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run Tests Verbosely
```bash
pytest -v
```

### Run Specific Test Functions
```bash
pytest tests/test_models.py::test_crop_model
pytest tests/test_model_accuracy.py::TestCropModelAccuracy::test_model_accuracy_threshold
```

## Test Fixtures

Available fixtures in `conftest.py`:

- `sample_crop_data`: Generated crop recommendation data (100 samples)
- `sample_fertilizer_data`: Generated fertilizer data (100 samples)
- `sample_yield_data`: Generated yield data (100 samples)
- `trained_crop_model`: Pre-trained crop model
- `preprocessor`: DataPreprocessor instance
- `temp_dir`: Temporary directory for test outputs
- `sample_input_features`: Sample input for predictions
- `engineered_features`: Pre-engineered features

## Writing New Tests

### Example Unit Test
```python
import pytest

@pytest.mark.unit
def test_my_function(sample_crop_data):
    """Test description."""
    # Arrange
    X = sample_crop_data.drop('label', axis=1)
    
    # Act
    result = my_function(X)
    
    # Assert
    assert result is not None
    assert len(result) == len(X)
```

### Example Model Test
```python
import pytest

@pytest.mark.model
def test_model_accuracy(trained_crop_model, sample_crop_data):
    """Test model meets accuracy threshold."""
    X = sample_crop_data.drop('label', axis=1)
    y = sample_crop_data['label']
    
    metrics = trained_crop_model.evaluate(X, y)
    assert metrics['accuracy'] > 0.8  # 80% accuracy threshold
```

## Test Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what is being tested
2. **One Assertion Per Test**: Prefer multiple small tests over one large test
3. **Use Fixtures**: Reuse common setup code via fixtures
4. **Mark Tests**: Use pytest markers to categorize tests
5. **Test Edge Cases**: Include tests for boundary conditions and error cases
6. **Keep Tests Fast**: Move slow tests to a separate category
7. **Test Independence**: Each test should be able to run independently

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=src --cov-report=xml
```

## Test Coverage Goals

- **Unit Tests**: > 80% code coverage
- **Model Tests**: All models tested for basic functionality
- **Integration Tests**: End-to-end workflows tested

## Debugging Failed Tests

### View Full Error Details
```bash
pytest -v --tb=long
```

### Run Failed Tests Only
```bash
pytest --lf  # last failed
pytest --ff  # failed first
```

### Enable Print Statements
```bash
pytest -s
```

### Use PDB Debugger
```bash
pytest --pdb  # Drop into debugger on failure
```

## Expected Test Results

When all tests pass, you should see output similar to:
```
=================== test session starts ===================
collected 45 items

tests/test_data_preprocessing.py ........            [ 17%]
tests/test_feature_engineering.py ...........        [ 42%]
tests/test_model_accuracy.py ................        [ 77%]
tests/test_models.py ...                             [ 84%]

=================== 45 passed in 12.34s ===================
```

## Troubleshooting

### Import Errors
Ensure the project root is in PYTHON PATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

### Fixture Not Found
Check that `conftest.py` is in the tests directory

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add new tests for new functionality
4. Update this README if adding new test categories

## Contact

For questions about tests or to report issues, please create a GitHub issue.
