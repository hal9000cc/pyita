"""Tests for CCI indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [2, 20])
def test_cci_vs_talib(test_ohlcv_data, period):
    """Test CCI calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates CCI using pyita
    4. Calculates CCI using TA-Lib
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
    cci_result = ta.cci(quotes, period=period)
    
    # Calculate with TA-Lib
    talib_cci = talib.CCI(high_data, low_data, close_data, timeperiod=period)
    talib_cci[np.isnan(talib_cci)] = 0
    
    # Compare CCI results
    assert arrays_equal_with_nan(
        cci_result.cci,
        talib_cci
    ), f"CCI (period={period}) does not match TA-Lib"

