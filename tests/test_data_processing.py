"""
Unit tests for data processing functions
"""

import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_and_clean_data, encode_categorical_features


class TestDataProcessing:
    """Test suite for data processing module"""
    
    def test_load_and_clean_data(self):
        """Test data loading and cleaning"""
        # Create sample data
        data = {
            'user_id': [1, 2, 3],
            'plan_type': ['Basic', 'Premium', 'Basic'],
            'monthly_fee': [500, 1000, 500],
            'churn': ['No', 'Yes', 'No']
        }
        df = pd.DataFrame(data)
        
        # Save to temp CSV
        temp_file = 'temp_test_data.csv'
        df.to_csv(temp_file, index=False)
        
        # Test loading
        result = load_and_clean_data(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'user_id' not in result.columns  # Should be dropped
        assert 'plan_type' in result.columns
    
    def test_encode_categorical_features(self):
        """Test categorical encoding"""
        data = {
            'plan_type': ['Basic', 'Premium', 'Basic'],
            'churn': ['No', 'Yes', 'No']
        }
        df = pd.DataFrame(data)
        
        encoded_df, encoders = encode_categorical_features(df)
        
        # Assertions
        assert isinstance(encoded_df, pd.DataFrame)
        assert isinstance(encoders, dict)
        assert 'plan_type' in encoders
        assert encoded_df['plan_type'].dtype in ['int64', 'int32']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
