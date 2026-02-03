"""Tests for RSI indicator."""
import numpy as np
import pytest
import py_ta as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [2, 5, 22, 12])
def test_rsi_vs_talib(test_ohlcv_data, period):
    """Test RSI calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates RSI using py-ta
    4. Calculates RSI using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: period.
    ma_type='mma' and value='close' are fixed.
    Note: TA-Lib RSI uses Wilder's smoothing (similar to MMA).
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement
    # RSI needs at least period + 1 values (one for diff, period for smoothing)
    data_length = len(close_data)
    assert data_length >= period + 1, f"Insufficient data: {data_length} bars, need at least {period + 1}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with py-ta (using default ma_type='mma')
    rsi_result = ta.rsi(quotes, period=period, ma_type='mma', value='close')
    
    # Calculate with TA-Lib
    # talib.RSI uses Wilder's smoothing (similar to MMA)
    talib_rsi = talib.RSI(close_data, timeperiod=period)
    
    # Compare RSI results
    py_ta_rsi = np.asarray(rsi_result.rsi)
    
    assert arrays_equal_with_nan(
        py_ta_rsi,
        talib_rsi
    ), f"RSI (period={period}) does not match TA-Lib"

