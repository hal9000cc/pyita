"""Pytest configuration and shared fixtures for py-ta tests."""
import pickle
from pathlib import Path

import numpy as np
import pytest

# Comparison tolerance for floating point comparisons
COMPARISON_TOLERANCE = 1e-12

# Fixed test data filename
TEST_DATA_FILENAME = "BINANCE_BTC_USDT_1h_2025.pkl"


def arrays_equal_with_nan(arr1, arr2, rtol=COMPARISON_TOLERANCE, atol=COMPARISON_TOLERANCE):
    """Compare two numpy arrays with tolerance, handling NaN values.
    
    Two arrays are considered equal if:
    - Their shapes match
    - For each position:
      - If both values are NaN, they are considered equal
      - If one is NaN and the other is not, they are not equal
      - If both are numbers, they are compared with numpy.isclose()
    
    Args:
        arr1: First numpy array
        arr2: Second numpy array
        rtol: Relative tolerance for numeric comparison (default: COMPARISON_TOLERANCE)
        atol: Absolute tolerance for numeric comparison (default: COMPARISON_TOLERANCE)
        
    Returns:
        bool: True if arrays are equal within tolerance, False otherwise
    """
    # Check shapes
    if arr1.shape != arr2.shape:
        return False
    
    # Create masks for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    # Check that NaN positions match
    if not np.array_equal(nan_mask1, nan_mask2):
        return False
    
    # For non-NaN positions, compare with isclose
    non_nan_mask = ~nan_mask1
    if np.any(non_nan_mask):
        return np.allclose(
            arr1[non_nan_mask],
            arr2[non_nan_mask],
            rtol=rtol,
            atol=atol,
            equal_nan=True
        )
    
    # If all values are NaN, arrays are equal
    return True


@pytest.fixture
def test_ohlcv_data():
    """Load OHLCV test data from pickle file.
    
    Returns:
        dict: Dictionary with keys 'time', 'open', 'high', 'low', 'close', 'volume'
            containing numpy arrays of OHLCV data
            
    Raises:
        AssertionError: If file not found or insufficient data
    """
    # Get test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    filepath = test_data_dir / TEST_DATA_FILENAME
    
    # Check if file exists
    assert filepath.exists(), f"Test data file not found: {filepath}"
    
    # Load data (now it's a dictionary with numpy arrays)
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Validate data structure
    assert isinstance(data_dict, dict), "Test data should be a dictionary"
    required_keys = ['time', 'open', 'high', 'low', 'close', 'volume']
    for key in required_keys:
        assert key in data_dict, f"Missing key '{key}' in test data"
    
    return data_dict

