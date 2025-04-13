# starter/test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_status_code():
    response = client.get("/")
    assert response.status_code == 200  # Check that the status code is 200 OK

def test_get_content():
    response = client.get("/")
    assert "Welcome" in response.text  # Replace this with actual expected content of your response

def test_model_prediction_positive():
    # Test case for a positive prediction (e.g., salary >50K)
    payload = {"age": 45, "education": "Bachelors", "hours-per-week": 50}  # Example input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # Ensure the status code is 200
    assert "prediction" in response.json()  # Check if the "prediction" field is in the response
    assert response.json()["prediction"] == 1  # Assuming 1 means salary >50K

def test_model_prediction_negative():
    # Test case for a negative prediction (e.g., salary <=50K)
    payload = {"age": 30, "education": "HS-grad", "hours-per-week": 40}  # Example input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # Ensure the status code is 200
    assert "prediction" in response.json()  # Check if the "prediction" field is in the response
    assert response.json()["prediction"] == 0  # Assuming 0 means salary <=50K
