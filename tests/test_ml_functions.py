"""
Unit tests for ML functions
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_functions import predict_churn


class TestMLFunctions:
    """Test suite for ML functions module"""
    
    def test_predict_churn_output_format(self):
        """Test that predict_churn returns correct format"""
        # Mock model for testing
        class MockModel:
            def predict(self, X):
                return np.array([1])
            
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])
        
        model = MockModel()
        features = [1, 500, 5.0, 2, 0, 12, 5]
        
        prediction, probability = predict_churn(model, features)
        
        # Assertions
        assert isinstance(prediction, (int, np.integer))
        assert isinstance(probability, (float, np.floating))
        assert 0 <= probability <= 1
        assert prediction in [0, 1]
    
    def test_predict_churn_probability_range(self):
        """Test that probability is in valid range"""
        class MockModel:
            def predict(self, X):
                return np.array([0])
            
            def predict_proba(self, X):
                return np.array([[0.8, 0.2]])
        
        model = MockModel()
        features = [0, 300, 10.0, 0, 0, 24, 2]
        
        _, probability = predict_churn(model, features)
        
        assert 0 <= probability <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
