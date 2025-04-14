import os

import joblib
import numpy as np
import pandas as pd

from .data import process_data


def load_model():
    """
    Load the trained model, encoder, and label binarizer.
    """
    # Get the current directory (which is .../starter/starter/ml)
    base_dir = os.path.dirname(__file__)

    # Build paths to model artifacts relative to the ml directory
    model_path = os.path.abspath(os.path.join(base_dir, "..", "model.joblib"))
    encoder_path = os.path.abspath(
        os.path.join(base_dir, "..", "encoder.joblib"))
    lb_path = os.path.abspath(os.path.join(
        base_dir, "..", "label_binarizer.joblib"))

    # Load the files
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    return model, encoder, lb


def model_inference(model, encoder, lb, data):
    """
    Perform inference (prediction) using the trained model.

    :param model: Trained model
    :param encoder: Pre-fitted encoder for processing categorical features
    :param lb: Pre-fitted label binarizer for target variable
    :param data: A dictionary containing all input features\
          (both continuous and categorical)
    :return: Prediction (in the same format as training labels)
    """
    # Convert the dictionary into a DataFrame (so it is 2D)
    df = pd.DataFrame([data])

    # Define the categorical features exactly as in training.
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the data. Note: label is None since we’re doing inference.
    X, _, _, _ = process_data(
        df,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict using the model
    prediction = model.predict(X)

    # Inverse transform to get the original label
    return lb.inverse_transform(prediction)[0]


def evaluate_model(test_data, model, encoder, lb, categorical_features):
    """
    Evaluate the model on test data and return performance metrics.

    :param test_data: DataFrame containing test data
    :param model: Trained model
    :param encoder: Pre-fitted encoder
    :param lb: Pre-fitted label binarizer
    :param categorical_features: List of categorical\
          feature names used during training
    :return: Accuracy as a float
    """
    # Pass encoder and lb to process_data when training is False
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    predictions = model.predict(X_test)
    predictions = lb.inverse_transform(predictions)
    accuracy = np.mean(predictions == y_test)
    return accuracy
