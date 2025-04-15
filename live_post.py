import requests

payload = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 123456,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

url = "https://ml-fastapi-deploy-ugra.onrender.com/predict"
response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Prediction:", response.json())
