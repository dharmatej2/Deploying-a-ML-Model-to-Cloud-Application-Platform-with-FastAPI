# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Import process_data from ml.data.py
from ml.data import process_data

# --- Path Handling ---
# Get the directory where this script resides
base_dir = os.path.dirname(__file__)

# Build absolute path to census_clean.csv inside /data folder
data_path = os.path.abspath(os.path.join(base_dir, "..", "data", "census_clean.csv"))
data = pd.read_csv(data_path)

# Split data into train and test
train, test = train_test_split(data, test_size=0.20)

# Define the categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Process the train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Save the model and encoders to the /starter directory
model_path = os.path.abspath(os.path.join(base_dir, "..", "model.joblib"))
encoder_path = os.path.abspath(os.path.join(base_dir, "..", "encoder.joblib"))
lb_path = os.path.abspath(os.path.join(base_dir, "..", "label_binarizer.joblib"))

joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)

print("Model training complete and saved.")