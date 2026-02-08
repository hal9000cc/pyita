"""Tests for ZigZag indicator."""
import numpy as np
import pytest
import py_ta as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('delta, type_param', [
    (0.02, 'high_low'),
    (0.01, 'high_low'),
    (0.01, 'close'),
])
def test_zigzag_vs_si(test_ohlcv_data, delta, type_param):
    """Test ZigZag calculation against stock-indicators reference.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates ZigZag using py-ta (with and without end_points)
    4. Verifies that ZigZag pivot types match stock-indicators
    5. Allows up to 2% difference due to algorithm differences
    
    Parameters are parametrized: delta, type_param.
    depth=1 and end_points=False are fixed.
    
    Note: Comparison starts from the 3rd pivot because the reference
    library calculates initial pivots slightly differently.
    """
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    # First call - just check it doesn't crash with end_points=True
    zigzag_with_endpoints = ta.zigzag(quotes, delta=delta, depth=1, type=type_param, end_points=True)
    
    # Second call - actual test
    zigzag = ta.zigzag(quotes, delta=delta, depth=1, type=type_param, end_points=False)

    # Map type to EndType string (converted to enum inside get_si_ref only when generating data)
    end_type_map = {
        'high_low': 'HIGH_LOW',
        'close': 'CLOSE',
        'open': 'CLOSE',
        'high': 'CLOSE',
        'low': 'CLOSE'
    }
    
    # Get reference values from stock-indicators (or from cache if available)
    # Pass string instead of enum - conversion happens inside get_si_ref only when needed
    ref = get_si_ref(TEST_DATA_FILENAME, 'get_zig_zag', end_type_map[type_param], delta * 100)

    # Convert reference point_type to numeric format
    ref_point_type = np.zeros(len(ref.point_type), dtype=np.int8)
    # Handle string values and None
    for i, val in enumerate(ref.point_type):
        if val == 'H':
            ref_point_type[i] = 1
        elif val == 'L':
            ref_point_type[i] = -1
        else:
            ref_point_type[i] = 0

    # Start checking from the 3rd pivot (index 2)
    non_zero_indices = np.flatnonzero(zigzag.pivot_types != 0)
    if len(non_zero_indices) >= 3:
        i_start_check = non_zero_indices[2]
        
        # Ensure arrays have same length for comparison
        min_len = min(len(zigzag.pivot_types) - i_start_check, len(ref_point_type) - i_start_check)
        
        # Calculate difference ratio
        diff_ratio = (zigzag.pivot_types[i_start_check:i_start_check + min_len] != ref_point_type[i_start_check:i_start_check + min_len]).sum() / min_len
        
        assert diff_ratio < 0.002, f"ZigZag (delta={delta}, type={type_param}) differs by {diff_ratio:.2%}, expected < 2%"

