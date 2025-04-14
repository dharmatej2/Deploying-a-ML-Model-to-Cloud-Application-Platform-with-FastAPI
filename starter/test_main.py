# starter/test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_status_code():
    response = client.get("/")
    assert response.status_code == 200  # Check that the status code is 200 OK


def test_get_content():
    response = client.get("/")
    assert (
        "Hello World" in response.text
    )  # Replace this with actual expected content of your response


def test_model_prediction_positive():
    # Test case for a positive prediction (e.g., salary >50K)
    payload = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States",
    }  # Example input
    response = client.post("/predict/", json=payload)
    print("Prediction result:", response.json())
    assert response.status_code == 200  # Ensure the status code is 200
    assert (
        "prediction" in response.json()
    )  # Check if the "prediction" field is in the response
    assert response.json()["prediction"] == " >50K"


def test_model_prediction_negative():
    # Test case for a negative prediction (e.g., salary <=50K)
    payload = {
        "age": 23,
        "workclass": "Private",
        "fnlgt": 0,
        "education": "HS-grad",
        "education_num": 0,
        "marital_status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 20,
        "native_country": "United-States",
    }  # Example input
    response = client.post("/predict/", json=payload)
    print("Prediction result:", response.json())
    assert response.status_code == 200  # Ensure the status code is 200
    assert (
        "prediction" in response.json()
    )  # Check if the "prediction" field is in the response
    assert response.json()["prediction"] == " <=50K"
