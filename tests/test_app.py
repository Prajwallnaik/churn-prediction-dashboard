"""
Integration tests for Streamlit app
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestApp:
    """Test suite for Streamlit application"""
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            import streamlit
            import pickle
            import numpy
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'models', 
            'best_classifier.pkl'
        )
        assert os.path.exists(model_path), "Model file not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
