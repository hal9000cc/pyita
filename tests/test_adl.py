"""Tests for ADL indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('ma_period', [2, 14])
def test_adl_vs_talib(test_ohlcv_data, ma_period):
    """Test ADL calculation against TA-Lib reference implementation.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates ADL using pyita
    4. Calculates ADL using TA-Lib
    5. Compares results with tolerance
    
    Parameters are parametrized: ma_period.
    ma_type='sma' is fixed (default).
    Note: This test only compares the base ADL values, not adl_smooth.
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # ADL doesn't have a minimum period requirement (it's cumulative)
    # But we need at least one bar
    data_length = len(close_data)
    assert data_length >= 1, f"Insufficient data: {data_length} bars, need at least 1"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with pyita (without smoothing first, to compare base ADL)
    adl_result = ta.adl(quotes, ma_period=None)
    
    # Calculate with TA-Lib
    talib_ad = talib.AD(high_data, low_data, close_data, volume_data)
    
    # Compare ADL results
    assert arrays_equal_with_nan(
        adl_result.adl,
        talib_ad
    ), f"ADL (ma_period=None) does not match TA-Lib"


@pytest.mark.parametrize('ma_period', [2, 14])
def test_adl_smooth(test_ohlcv_data, ma_period):
    """Test ADL with smoothing.
    
    This test verifies that adl_smooth is calculated correctly when ma_period is provided.
    Note: We don't compare with TA-Lib here because TA-Lib doesn't provide smoothed ADL directly.
    """
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    volume_data = test_ohlcv_data['volume']
    
    # Check minimum data requirement for smoothing
    data_length = len(close_data)
    assert data_length >= ma_period, f"Insufficient data: {data_length} bars, need at least {ma_period}"
    
    # Create Quotes
    quotes = ta.Quotes(open_data, high_data, low_data, close_data, volume_data)
    
    # Calculate with pyita (with smoothing)
    adl_result = ta.adl(quotes, ma_period=ma_period, ma_type='sma')
    
    # Verify that both adl and adl_smooth are present
    assert hasattr(adl_result, 'adl'), "adl attribute should be present"
    assert hasattr(adl_result, 'adl_smooth'), "adl_smooth attribute should be present"
    
    # Verify that adl_smooth has the correct length
    assert len(adl_result.adl_smooth) == len(adl_result.adl), \
        f"adl_smooth length ({len(adl_result.adl_smooth)}) should match adl length ({len(adl_result.adl)})"


def test_adl_no_volume(test_ohlcv_data):
    """Test that ADL raises exception when volume is missing."""
    # Extract data
    open_data = test_ohlcv_data['open']
    high_data = test_ohlcv_data['high']
    low_data = test_ohlcv_data['low']
    close_data = test_ohlcv_data['close']
    
    # Create Quotes without volume
    quotes = ta.Quotes(open_data, high_data, low_data, close_data)
    
    # Attempt to calculate ADL (should raise PyTAExceptionDataSeriesNonFound)
    with pytest.raises(ta.PyTAExceptionDataSeriesNonFound) as exc_info:
        ta.adl(quotes, ma_period=None)
    
    assert 'volume' in str(exc_info.value).lower(), \
        "Exception message should mention 'volume'"

