"""
Data Processing Functions
Handles data loading, cleaning, and preprocessing for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os


def load_and_clean_data(file_path):
    """
    Load and clean customer churn data
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(columns=["user_id", "signup_date"], errors="ignore")
    
    # Handle missing values
    df = df.dropna()
    
    return df


def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features using LabelEncoder
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of column names to encode
    
    Returns:
        Encoded DataFrame and fitted encoders
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    return df_encoded, encoders


def create_preprocessing_pipeline(df, target_col='churn'):
    """
    Create complete preprocessing pipeline
    
    Args:
        df: Raw DataFrame
        target_col: Name of target column
    
    Returns:
        Processed features, target, and pipeline artifacts
    """
    # Clean data
    df_clean = load_and_clean_data(df) if isinstance(df, str) else df.copy()
    
    # Separate features and target
    if target_col in df_clean.columns:
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
    else:
        X = df_clean
        y = None
    
    # Encode categorical features
    X_encoded, encoders = encode_categorical_features(X)
    
    # Encode target if exists
    if y is not None and y.dtype == 'object':
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        encoders['target'] = le_target
    else:
        y_encoded = y
    
    return X_encoded, y_encoded, encoders


def save_preprocessing_artifacts(encoders, scaler=None, output_dir="../models"):
    """
    Save preprocessing artifacts (encoders, scalers) to disk
    
    Args:
        encoders: Dictionary of fitted encoders
        scaler: Fitted scaler (optional)
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save label encoders
    pickle.dump(encoders, open(os.path.join(output_dir, "label_encoder.pkl"), "wb"))
    
    # Save scaler if provided
    if scaler is not None:
        pickle.dump(scaler, open(os.path.join(output_dir, "data_processing_pipeline.pkl"), "wb"))
    
    print(f"Preprocessing artifacts saved to {output_dir}")


def load_preprocessing_artifacts(artifact_dir="../models"):
    """
    Load preprocessing artifacts from disk
    
    Args:
        artifact_dir: Directory containing artifacts
    
    Returns:
        Dictionary of loaded artifacts
    """
    artifacts = {}
    
    encoder_path = os.path.join(artifact_dir, "label_encoder.pkl")
    if os.path.exists(encoder_path):
        artifacts['encoders'] = pickle.load(open(encoder_path, "rb"))
    
    pipeline_path = os.path.join(artifact_dir, "data_processing_pipeline.pkl")
    if os.path.exists(pipeline_path):
        artifacts['pipeline'] = pickle.load(open(pipeline_path, "rb"))
    
    return artifacts
