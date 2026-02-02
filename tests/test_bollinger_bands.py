"""Tests for Bollinger Bands indicator."""
import numpy as np
import py_ta as ta
import talib

from conftest import arrays_equal_with_nan, COMPARISON_TOLERANCE


def test_bollinger_bands_vs_talib(test_ohlcv_data):
    """Test Bollinger Bands calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates Bollinger Bands using py-ta
    4. Calculates Bollinger Bands using TA-Lib
    5. Compares results with tolerance
    
    Parameters:
        period=200, deviation=3, ma_type='sma', value='close'
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement for period=200
    data_length = len(close_data)
    assert data_length >= 200, f"Insufficient data: {data_length} bars, need at least 200"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with py-ta
    bb = ta.bollinger_bands(quotes, period=200, deviation=3, ma_type='sma', value='close')
    
    # Calculate with TA-Lib
    # talib.BBANDS(close, timeperiod=200, nbdevup=3, nbdevdn=3, matype=0)
    # matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
    talib_upper, talib_middle, talib_lower = talib.BBANDS(
        close_data,
        timeperiod=200,
        nbdevup=3,
        nbdevdn=3,
        matype=0  # SMA
    )
    
    # Compare results
    # Convert to numpy arrays if needed
    py_ta_mid = np.asarray(bb.mid_line)
    py_ta_upper = np.asarray(bb.up_line)
    py_ta_lower = np.asarray(bb.down_line)
    
    # Compare middle line
    assert arrays_equal_with_nan(
        py_ta_mid,
        talib_middle,
        rtol=COMPARISON_TOLERANCE,
        atol=COMPARISON_TOLERANCE
    ), "Middle line (SMA) does not match TA-Lib"
    
    # Compare upper band
    assert arrays_equal_with_nan(
        py_ta_upper,
        talib_upper,
        rtol=COMPARISON_TOLERANCE,
        atol=COMPARISON_TOLERANCE
    ), "Upper band does not match TA-Lib"
    
    # Compare lower band
    assert arrays_equal_with_nan(
        py_ta_lower,
        talib_lower,
        rtol=COMPARISON_TOLERANCE,
        atol=COMPARISON_TOLERANCE
    ), "Lower band does not match TA-Lib"

