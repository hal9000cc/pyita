"""Tests for ROC indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [2, 14])
def test_roc_vs_talib(test_ohlcv_data, period):
    """Test ROC calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates ROC using pyita
    4. Calculates ROC using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: period.
    ma_period=period and ma_type='sma' are fixed (defaults).
    value='close' is fixed (default).
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
    
    # Calculate with pyita (using default ma_period=period, ma_type='sma')
    roc_result = ta.roc(quotes, period=period, ma_period=period, ma_type='sma', value='close')
    
    # Calculate with TA-Lib
    talib_roc = talib.ROC(close_data, timeperiod=period)
    
    # Compare ROC results
    assert arrays_equal_with_nan(
        roc_result.roc,
        talib_roc
    ), f"ROC (period={period}) does not match TA-Lib"

