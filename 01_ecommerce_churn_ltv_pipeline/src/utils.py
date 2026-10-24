import os
import pickle
import numpy as np

def save_object(file_path: str, obj):
    """Saves a python object as a pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def load_object(file_path: str):
    """Loads a python object from a pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) to detect distribution drift.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift (triggers model retraining alert)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division with small epsilon
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)
