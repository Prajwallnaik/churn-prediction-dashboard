"""
Main Entry Point for Churn Prediction Application
Run this script to start the Streamlit application
"""

import os
import sys

# Add Scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import subprocess
    
    # Get the path to app.py
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Run Streamlit
    subprocess.run(["streamlit", "run", app_path])
