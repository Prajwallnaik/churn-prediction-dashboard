"""
Helper Functions
Utility functions for the churn prediction application
"""

import os
import json
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_artifact_path(filename):
    """
    Get full path to artifact file in models directory
    
    Args:
        filename: Name of artifact file
    
    Returns:
        Full path to artifact
    """
    return os.path.join(get_project_root(), "models", filename)


def get_data_path(filename):
    """
    Get full path to data file
    
    Args:
        filename: Name of data file
    
    Returns:
        Full path to data file
    """
    return os.path.join(get_project_root(), "data", filename)


def log_prediction(customer_data, prediction, probability, log_file="prediction_log.json"):
    """
    Log prediction results to JSON file
    
    Args:
        customer_data: Dictionary of customer features
        prediction: Churn prediction (0 or 1)
        probability: Churn probability
        log_file: Path to log file
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_data": customer_data,
        "prediction": int(prediction),
        "probability": float(probability)
    }
    
    log_path = get_artifact_path(log_file)
    
    # Load existing logs
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new log
    logs.append(log_entry)
    
    # Save logs
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)


def format_currency(amount, currency="₹"):
    """
    Format amount as currency
    
    Args:
        amount: Numeric amount
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    return f"{currency}{amount:,.2f}"


def calculate_risk_level(probability):
    """
    Calculate risk level from churn probability
    
    Args:
        probability: Churn probability (0-1)
    
    Returns:
        Risk level string
    """
    if probability >= 0.75:
        return "Critical"
    elif probability >= 0.5:
        return "High"
    elif probability >= 0.25:
        return "Moderate"
    else:
        return "Low"


def get_retention_recommendations(probability, customer_data):
    """
    Generate retention recommendations based on churn probability and customer data
    
    Args:
        probability: Churn probability
        customer_data: Dictionary of customer features
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if probability >= 0.5:
        if customer_data.get('support_tickets', 0) > 3:
            recommendations.append("Prioritize resolving outstanding support issues")
        
        if customer_data.get('payment_failures', 0) > 0:
            recommendations.append("Contact customer about payment issues")
        
        if customer_data.get('avg_weekly_usage_hours', 0) < 2:
            recommendations.append("Engage customer with usage tutorials or onboarding")
        
        if customer_data.get('tenure_months', 0) < 6:
            recommendations.append("Offer new customer retention discount")
        
        recommendations.append("Consider personalized retention offer")
    
    return recommendations if recommendations else ["Monitor customer engagement"]
