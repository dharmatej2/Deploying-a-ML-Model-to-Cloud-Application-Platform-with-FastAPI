from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from starter.starter.ml.model import load_model  # We will load our model, encoder, lb from here
from starter.starter.ml.data import process_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create the FastAPI app instance
app = FastAPI()

# Load the model, encoder, and label binarizer once at startup
model, encoder, lb = load_model()

# Define the schema for input data using Pydantic
class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def load_model_and_encoder():
    """
    On FastAPI startup, load the model and encoder.
    This is called when the server starts.
    """
    global model, encoder, lb
    model, encoder, lb = load_model()

@app.post("/predict/")
async def predict(input_data: PredictionInput):
    """
    Predict the salary class (>50K or <=50K) based on input data
    """
    # Convert the incoming data to a DataFrame, similar to your training
    data = input_data.dict()
    
    # Define the categorical features list (same as during training)
    categorical_features = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]
    
    # Process the data like you did for training (no need to worry about 'label')
    X, _, _, _ = process_data(
        pd.DataFrame([data]),  # Convert the single input dict to a DataFrame
        categorical_features=categorical_features,
        label=None,  # We're just predicting, not training
        training=False,  # We're not in training mode
        encoder=encoder,
        lb=lb
    )
    
    # Make a prediction using the loaded model
    prediction = model.predict(X)
    
    # Convert the prediction back to its original label
    prediction_label = lb.inverse_transform(prediction)[0]
    
    # Return the result as a JSON response
    return {"prediction": prediction_label}
