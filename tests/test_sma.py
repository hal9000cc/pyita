"""Tests for SMA indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [2, 5, 10, 20, 50, 100, 300, 500])
def test_sma_vs_talib(test_ohlcv_data, period):
    """Test SMA calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates SMA using pyita
    4. Calculates SMA using TA-Lib
    5. Compares results with tolerance
    
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
    sma_result = ta.sma(quotes, period=period, value='close')
    
    # Calculate with TA-Lib
    talib_sma = talib.SMA(close_data, timeperiod=period)
    
    # Compare results
    assert arrays_equal_with_nan(
        sma_result.sma,
        talib_sma
    ), f"SMA (period={period}) does not match TA-Lib"


@pytest.mark.parametrize('period', [1, 3, 5, 22])
def test_sma_direct_calculation(test_ohlcv_data, period):
    """Test SMA calculation by direct computation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates SMA using pyita
    4. Verifies that SMA values match direct arithmetic mean calculation
    
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
    sma_result = ta.sma(quotes, period=period, value='close')
    
    # Get values
    values = close_data
    
    # Calculate expected SMA by direct computation
    expected_sma = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(period - 1, len(values)):
        expected_sma[i] = values[i - period + 1: i + 1].sum() / period
    
    # Compare using arrays_equal_with_nan
    assert arrays_equal_with_nan(
        sma_result.sma,
        expected_sma
    ), f"SMA (period={period}) does not match direct calculation"

