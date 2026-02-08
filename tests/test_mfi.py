"""Tests for MFI indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [2, 3, 20])
def test_mfi_vs_talib(test_ohlcv_data, period):
    """Test MFI calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates MFI using pyita
    4. Calculates MFI using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: period.
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
    mfi_result = ta.mfi(quotes, period=period)
    
    # Calculate with TA-Lib
    talib_mfi = talib.MFI(high_data, low_data, close_data, volume_data, timeperiod=period)
    
    # Compare MFI results
    assert arrays_equal_with_nan(
        mfi_result.mfi,
        talib_mfi
    ), f"MFI (period={period}) does not match TA-Lib"

