import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .data import process_data
from .model import evaluate_model, load_model, model_inference


# Helper to get the absolute path to data
def get_data_path():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(
        os.path.join(current_dir, "..", "..", "data", "census_clean.csv")
    )


# Test for the load_model function
def test_load_model():
    model, encoder, lb = load_model()

    # Check that the model is of the correct type
    assert model is not None
    assert hasattr(model, "predict")

    # Check that encoder and label binarizer are not None
    assert encoder is not None
    assert hasattr(encoder, "transform")

    assert lb is not None
    assert hasattr(lb, "transform")


# Test for the model_inference function
def test_model_inference():
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
        "native-country": "United-States",
    }

    model, encoder, lb = load_model()
    result = model_inference(model, encoder, lb, data)
    assert result in [" <=50K", " >50K"]


# Test for the evaluate_model function
def test_evaluate_model():
    data_path = get_data_path()
    data = pd.read_csv(data_path)

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)

    accuracy = evaluate_model(test, model, encoder, lb, cat_features)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
