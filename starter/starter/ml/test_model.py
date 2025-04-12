import pytest
import joblib
import numpy as np
from .model import load_model, model_inference, evaluate_model
from .data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Test for the load_model function
def test_load_model():
    model, encoder, lb = load_model()
    
    # Check that the model is of the correct type
    assert model is not None
    assert hasattr(model, 'predict')  # Ensure it's a model with a predict method
    
    # Check that encoder and label binarizer are not None
    assert encoder is not None
    assert hasattr(encoder, 'transform')  # Ensure it's an encoder
    
    assert lb is not None
    assert hasattr(lb, 'transform')  # Ensure it's a label binarizer

# Test for the model_inference function
def test_model_inference():
    # Create a dummy data sample with both continuous and categorical features.
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    # Load the model, encoder, and label binarizer
    model, encoder, lb = load_model()
    
    # Perform inference using the updated model_inference function
    result = model_inference(model, encoder, lb, data)
    
    # Check that the result is one of the expected classes
    assert result in [' <=50K', ' >50K']



# Test for the evaluate_model function
def test_evaluate_model():
    # Load the test data (assuming it's in the correct path)
    data_path = "/home/dharmatej/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/starter/data/census_clean.csv"
    data = pd.read_csv(data_path)
    
    # Split data into train and test
    train, test = train_test_split(data, test_size=0.20)
    
    # Process the data
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Train the model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    accuracy = evaluate_model(test, model, encoder, lb, cat_features)
    
    # Check that accuracy is a float between 0 and 1
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


