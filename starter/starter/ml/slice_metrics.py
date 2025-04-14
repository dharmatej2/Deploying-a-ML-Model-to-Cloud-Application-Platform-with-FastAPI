import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data
from .model import load_model


def compute_slice_metrics(data: pd.DataFrame,\
                           slice_column: str, output_file: str):
    # Dynamically construct model paths
    base_path = Path(__file__).resolve().parent.parent
    base_path / "model.joblib"
    base_path / "encoder.joblib"
    base_path / "label_binarizer.joblib"

    model, encoder, lb = load_model()

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

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for value in data[slice_column].unique():
            slice_data = data[data[slice_column] == value]
            X, y, _, _ = process_data(
                slice_data,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = model.predict(X)
            precision = precision_score(y, preds)
            recall = recall_score(y, preds)
            fbeta = fbeta_score(y, preds, beta=1.0)

            f.write(
                f"Feature: {slice_column} = {value} | "
                f"Precision: {precision:.4f} | Recall:\
                      {recall:.4f} | F1: {fbeta:.4f}\n"
            )
