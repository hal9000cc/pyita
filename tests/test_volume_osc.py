"""Tests for Volume Oscillator indicator."""
import numpy as np
import pytest
import py_ta as ta

from conftest import arrays_equal_with_nan
from py_ta.move_average import ma_calculate, MA_Type


@pytest.mark.parametrize('period_short', [
    1,
    3,
    5,
    22,
    22,
])
def test_volume_osc_direct_calculation(test_ohlcv_data, period_short):
    """Test Volume Oscillator calculation by direct computation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates Volume Oscillator using py-ta
    4. Verifies that Volume Oscillator values match direct calculation:
       - vol_short = MA(volume, period_short)
       - vol_long = MA(volume, period_long)
       - osc = (vol_short - vol_long) / vol_long * 100
    
    Parameters are parametrized: period_short.
    period_long=100 is fixed.
    ma_type='ema' is fixed.
    """
    period_long = 100
    
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement
    data_length = len(volume_data)
    assert data_length >= period_long, f"Insufficient data: {data_length} bars, need at least {period_long}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with py-ta
    vosc_result = ta.volume_osc(quotes, period_short=period_short, period_long=period_long, ma_type='ema')
    
    # Calculate expected values by direct computation
    vshort = ma_calculate(volume_data, period_short, MA_Type.ema)
    vlong = ma_calculate(volume_data, period_long, MA_Type.ema)
    
    np.seterr(divide='ignore', invalid='ignore')
    expected_osc = (vshort - vlong) / vlong * 100
    
    # Compare using arrays_equal_with_nan
    assert arrays_equal_with_nan(
        vosc_result.osc,
        expected_osc
    ), f"Volume Oscillator (period_short={period_short}, period_long={period_long}) does not match direct calculation"

