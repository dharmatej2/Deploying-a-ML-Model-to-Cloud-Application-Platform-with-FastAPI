import joblib
import numpy as np
import pandas as pd
from .data import process_data


def load_model():
    """
    Load the trained model, encoder, and label binarizer.
    """
    model = joblib.load('/home/dharmatej/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/starter/starter/model.joblib')  # load the trained model
    encoder = joblib.load('/home/dharmatej/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/starter/starter/encoder.joblib')  # load the encoder (for categorical features)
    lb = joblib.load('/home/dharmatej/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/starter/starter/label_binarizer.joblib')  # load the label binarizer (for target labels)
    
    return model, encoder, lb

def model_inference(model, encoder, lb, data):
    """
    Perform inference (prediction) using the trained model.
    
    :param model: Trained model
    :param encoder: Pre-fitted encoder for processing categorical features
    :param lb: Pre-fitted label binarizer for target variable
    :param data: A dictionary containing all input features (both continuous and categorical)
    :return: Prediction (in the same format as training labels)
    """
    # Convert the dictionary into a DataFrame (so it is 2D)
    df = pd.DataFrame([data])
    
    # Define the categorical features exactly as in training.
    categorical_features = ["workclass", "education", "marital-status", "occupation",
                            "relationship", "race", "sex", "native-country"]
    
    # Process the data. Note: label is None since we’re doing inference.
    X, _, _, _ = process_data(df, categorical_features=categorical_features, label=None,
                                training=False, encoder=encoder, lb=lb)
    
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
    :param categorical_features: List of categorical feature names used during training
    :return: Accuracy as a float
    """
    # Pass encoder and lb to process_data when training is False
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )
    predictions = model.predict(X_test)
    predictions = lb.inverse_transform(predictions)
    accuracy = np.mean(predictions == y_test)
    return accuracy

