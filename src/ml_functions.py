"""
Machine Learning Functions for Churn Prediction
Contains model training, evaluation, and prediction utilities
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import os


def train_churn_model(data_path, output_path="../models/best_classifier.pkl"):
    """
    Train XGBoost model with GridSearch optimization
    
    Args:
        data_path: Path to CSV data file
        output_path: Path to save trained model
    
    Returns:
        Trained model and performance metrics
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    df = df.drop(columns=["user_id", "signup_date"], errors="ignore")
    
    # Encode categorical variables
    le = LabelEncoder()
    df["plan_type"] = le.fit_transform(df["plan_type"])
    df["churn"] = le.fit_transform(df["churn"])
    
    # Split features & target
    X = df.drop("churn", axis=1)
    y = df["churn"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Base model
    xgb = XGBClassifier(eval_metric="logloss")
    
    # Grid parameters
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1],
    }
    
    # GridSearch
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Train
    print("Training model with GridSearch...")
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predict
    y_pred = best_model.predict(X_test)
    
    # Results
    print("\nBest Parameters:\n", grid_search.best_params_)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(best_model, open(output_path, "wb"))
    print(f"\nModel saved to {output_path}")
    
    return best_model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "best_params": grid_search.best_params_
    }


def load_model(model_path="../models/best_classifier.pkl"):
    """Load trained model from pickle file"""
    return pickle.load(open(model_path, "rb"))


def predict_churn(model, features):
    """
    Predict churn for given features
    
    Args:
        model: Trained model
        features: Array of features [plan_type, monthly_fee, usage, tickets, failures, tenure, last_login]
    
    Returns:
        Prediction (0/1) and probability
    """
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    
    return prediction, probability
