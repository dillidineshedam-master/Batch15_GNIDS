import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.logging_config import get_logger

logger = get_logger("DataLoader")

class LogStandardScaler:
    """Applies log1p compression to tame heavy-tailed flow bursts followed by z-score standardization."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_log = np.log1p(np.maximum(X, 0.0))
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        X_log = np.log1p(np.maximum(X, 0.0))
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def load_and_validate_dataset(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    file_path = cfg["data"]["raw_file"]
    feature_cols = cfg["data"]["features"]

    if not os.path.exists(file_path):
        logger.error(f"Dataset file '{file_path}' not found!")
        raise FileNotFoundError(f"Missing {file_path}. Place 'unsw_nb15_benchmark.csv' in 'data/'.")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} total flow records from {file_path}")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "label" not in df.columns and "Class" in df.columns:
        df["label"] = df["Class"].astype(int)
    elif "label" not in df.columns and "target" in df.columns:
        df["label"] = df["target"].astype(int)
    else:
        df["label"] = df["label"].astype(int)

    normal_mask = (df["label"] == 0)
    scaler = LogStandardScaler()
    scaler.fit(df.loc[normal_mask, feature_cols].values)
    logger.info(f"Log-Standardized baseline established on {normal_mask.sum():,} normal flows.")

    return df, scaler, feature_cols, cfg

if __name__ == "__main__":
    df, scaler, features, _ = load_and_validate_dataset()
    print(f"Data Loader Test Passed: {df.shape[0]} Rows, {len(features)} Features.")
