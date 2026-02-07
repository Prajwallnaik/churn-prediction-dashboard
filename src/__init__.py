"""
Churn Prediction Package
MLOps-compliant structure for customer churn prediction
"""

__version__ = "1.0.0"
__author__ = "Churn Intelligence Team"

# Package-level imports for convenience
from .ml_functions import train_churn_model, load_model, predict_churn
from .data_processing import load_and_clean_data, encode_categorical_features
from .helper_functions import get_project_root, calculate_risk_level

__all__ = [
    "train_churn_model",
    "load_model",
    "predict_churn",
    "load_and_clean_data",
    "encode_categorical_features",
    "get_project_root",
    "calculate_risk_level",
]
