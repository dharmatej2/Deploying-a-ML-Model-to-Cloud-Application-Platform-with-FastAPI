import pandas as pd
from ml.slice_metrics import compute_slice_metrics
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/census_clean.csv")
    df = pd.read_csv(data_path)

    output_file = os.path.join(current_dir, "../../slice_output.txt")
    compute_slice_metrics(df, slice_column="education", output_file=output_file)
    