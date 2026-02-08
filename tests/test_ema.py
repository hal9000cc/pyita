"""Tests for EMA indicator."""
import numpy as np
import pytest
import pyita as ta

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [1, 3, 5, 22])
def test_ema_direct_calculation(test_ohlcv_data, period):
    """Test EMA calculation by direct computation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates EMA using pyita
    4. Verifies that EMA values match direct calculation:
       - alpha = 2.0 / (period + 1)
       - First value (at index period-1) is SMA of first period elements
       - Subsequent values: ema[i] = source[i] * alpha + ema[i-1] * (1 - alpha)
    
    Parameters are parametrized: period.
    value='close' is fixed.
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement
    data_length = len(close_data)
    assert data_length >= period, f"Insufficient data: {data_length} bars, need at least {period}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with pyita
    ema_result = ta.ema(quotes, period=period, value='close')
    
    # Get values
    values = close_data
    
    # Calculate expected EMA by direct computation
    alpha = 2.0 / (period + 1)
    alpha_n = 1.0 - alpha
    
    # Find first non-NaN value
    start = 0
    for i, value in enumerate(values):
        if not np.isnan(value):
            start = i
            break
    
    # Check we have enough data
    assert len(values) >= start + period, f"Insufficient data after skipping NaNs"
    
    # Initialize result array
    expected_ema = np.full(len(values), np.nan, dtype=np.float64)
    
    # First period-1 elements are NaN
    # Element at index (start + period - 1) is SMA of first period elements
    first_index = start + period - 1
    first_value = values[start:start + period].sum() / period
    expected_ema[first_index] = first_value
    
    # Calculate subsequent EMA values
    # Note: if source value is NaN, EMA becomes NaN and stays NaN
    ema_value = first_value
    for i in range(first_index + 1, len(values)):
        if np.isnan(values[i]):
            ema_value = np.nan
        else:
            ema_value = values[i] * alpha + ema_value * alpha_n
        expected_ema[i] = ema_value
    
    # Compare using arrays_equal_with_nan
    assert arrays_equal_with_nan(
        ema_result.ema,
        expected_ema
    ), f"EMA (period={period}) does not match direct calculation"
