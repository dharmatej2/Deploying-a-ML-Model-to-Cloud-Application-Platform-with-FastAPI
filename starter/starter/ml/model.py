import joblib
import numpy as np
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
    :param data: Input data for which predictions are to be made
    :return: Prediction (in the same format as training labels)
    """
    # Preprocess the input data (this assumes data is a dictionary)
    # Extract values in the same order as during training
    ordered_keys = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    data_values = [data[key] for key in ordered_keys]
    processed_data = encoder.transform([data_values])

    prediction = model.predict(processed_data)  # Make the prediction
    
    # Return the predicted label (e.g., " <=50K" or " >50K")
    return lb.inverse_transform(prediction)[0]  # Inverse transform to get original label

def evaluate_model(test_data, model, encoder, lb, categorical_features):
    """
    Evaluate the model on test data and return performance metrics.

    :param test_data: Dataframe containing test data
    :param model: Trained model
    :param encoder: Pre-fitted encoder
    :param lb: Pre-fitted label binarizer
    :return: Accuracy or any other metric
    """
    # Preprocess the test data
    X_test, y_test, _, _ = process_data(test_data, categorical_features, label='salary', training=False)

    # Predict on the test data
    predictions = model.predict(X_test)
    
    # Convert predictions back to original labels
    predictions = lb.inverse_transform(predictions)
    
    # Evaluate accuracy (you can extend this to more metrics)
    accuracy = np.mean(predictions == y_test)
    return accuracy
