"""Tests for MACD indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period_short, period_long, period_signal', [
    (2, 3, 9),
    (2, 3, 2),
    (14, 21, 3),
    (8, 14, 9),
])
def test_macd_vs_talib(test_ohlcv_data, period_short, period_long, period_signal):
    """Test MACD calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates MACD using pyita
    4. Calculates MACD using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: period_short, period_long, period_signal.
    ma_type='ema' and ma_type_signal='sma' are fixed (defaults).
    value='close' is fixed (default).
    Note: TA-Lib MACD uses EMA for MACD lines and EMA for signal line.
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement
    data_length = len(close_data)
    assert data_length >= period_long, f"Insufficient data: {data_length} bars, need at least {period_long}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with pyita (using default ma_type='ema', ma_type_signal='sma')
    macd_result = ta.macd(quotes, period_short=period_short, period_long=period_long, 
                          period_signal=period_signal, ma_type='sma', ma_type_signal='emaw', value='close')
    
    talib_macd, talib_signal, talib_hist = talib.MACDEXT(
        close_data,
        fastperiod=period_short,
        slowperiod=period_long,
        signalperiod=period_signal,
         fastmatype=0,
         slowmatype=0,  
         signalmatype=1)
    
    # Compare MACD line results
    assert arrays_equal_with_nan(
        macd_result.macd[period_signal + 1],
        talib_macd[period_signal + 1]
    ), f"MACD line (period_short={period_short}, period_long={period_long}) does not match TA-Lib"
    
    # Compare Signal line results
    assert arrays_equal_with_nan(
        macd_result.signal[period_signal + 1],
        talib_signal[period_signal + 1]
    ), f"Signal line (period_signal={period_signal}) does not match TA-Lib"
    
    # Compare Histogram results
    assert arrays_equal_with_nan(
        macd_result.hist[period_signal + 1],
        talib_hist[period_signal + 1]
    ), f"Histogram does not match TA-Lib"

