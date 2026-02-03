"""Tests for ATR indicator."""
import numpy as np
import pytest
import py_ta as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('smooth', [2, 14])
def test_atr_vs_talib(test_ohlcv_data, smooth):
    """Test ATR calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates ATR using py-ta
    4. Calculates ATR using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: smooth.
    ma_type='mma' is fixed (default).
    Note: TA-Lib ATR uses Wilder's smoothing (similar to MMA).
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement
    data_length = len(close_data)
    assert data_length >= smooth, f"Insufficient data: {data_length} bars, need at least {smooth}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with py-ta (using default ma_type='mma')
    atr_result = ta.atr(quotes, smooth=smooth, ma_type='mma')
    
    # Calculate with TA-Lib
    # talib.ATR uses Wilder's smoothing (similar to MMA)
    talib_atr = talib.ATR(high_data, low_data, close_data, timeperiod=smooth)
    
    # Compare ATR results
    py_ta_atr = np.asarray(atr_result.atr)
    
    assert arrays_equal_with_nan(
        py_ta_atr,
        talib_atr
    ), f"ATR (smooth={smooth}) does not match TA-Lib"

