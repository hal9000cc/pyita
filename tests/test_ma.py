"""Tests for MA indicator."""
import numpy as np
import pytest
import pyita as ta
import talib

from conftest import arrays_equal_with_nan


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_sma_vs_sma_indicator(test_ohlcv_data, period):
    """Test MA with ma_type='sma' against ta.sma indicator."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='sma')
    
    # Calculate with SMA indicator
    sma_result = ta.sma(quotes, period=period, value='close')
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        sma_result.sma
    ), f"MA (ma_type='sma', period={period}) does not match SMA indicator"


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_ema_vs_ema_indicator(test_ohlcv_data, period):
    """Test MA with ma_type='ema' against ta.ema indicator."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='ema')
    
    # Calculate with EMA indicator
    ema_result = ta.ema(quotes, period=period, value='close')
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        ema_result.ema
    ), f"MA (ma_type='ema', period={period}) does not match EMA indicator"


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_ema_direct_calculation(test_ohlcv_data, period):
    """Test MA with ma_type='ema' by direct calculation."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='ema')
    
    # Calculate expected EMA by direct computation
    # EMA uses iema_calculate: initialization via SMA of first period elements starting from first non-NaN
    source_values = close_data
    expected_ema = np.full(len(source_values), np.nan, dtype=np.float64)
    
    if len(source_values) >= period:
        # Find first non-NaN value
        start_idx = 0
        for i, val in enumerate(source_values):
            if not np.isnan(val):
                start_idx = i
                break
        
        # Calculate initial SMA (first period elements starting from start_idx)
        if start_idx + period <= len(source_values):
            initial_sma = source_values[start_idx:start_idx + period].sum() / period
            # First value is set at start_idx + period - 1
            expected_ema[start_idx + period - 1] = initial_sma
            
            # Calculate EMA with alpha = 2.0 / (period + 1)
            alpha = 2.0 / (period + 1)
            alpha_n = 1.0 - alpha
            ema_value = initial_sma
            
            for i in range(start_idx + period, len(source_values)):
                ema_value = source_values[i] * alpha + ema_value * alpha_n
                expected_ema[i] = ema_value
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        expected_ema
    ), f"MA (ma_type='ema', period={period}) does not match direct calculation"


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_mma0_direct_calculation(test_ohlcv_data, period):
    """Test MA with ma_type='mma0' by direct calculation."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='mma0')
    
    # Calculate expected MMA0 by direct computation
    source_values = close_data
    expected_mma0 = np.full(len(source_values), np.nan, dtype=np.float64)
    
    # MMA0 initialization: first value is first data element
    if len(source_values) > 0:
        # Find first non-NaN value
        start_idx = 0
        for i, val in enumerate(source_values):
            if not np.isnan(val):
                start_idx = i
                break
        else:
            start_idx = len(source_values)
        
        if start_idx < len(source_values):
            # Initialize with first value
            ema_value = source_values[start_idx]
            expected_mma0[start_idx] = ema_value
            
            # Calculate MMA0 with alpha = 1.0 / period
            alpha = 1.0 / period
            alpha_n = 1.0 - alpha
            
            for i in range(start_idx + 1, len(source_values)):
                ema_value = source_values[i] * alpha + ema_value * alpha_n
                expected_mma0[i] = ema_value
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        expected_mma0
    ), f"MA (ma_type='mma0', period={period}) does not match direct calculation"


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_ema0_direct_calculation(test_ohlcv_data, period):
    """Test MA with ma_type='ema0' by direct calculation."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='ema0')
    
    # Calculate expected EMA0 by direct computation
    source_values = close_data
    expected_ema0 = np.full(len(source_values), np.nan, dtype=np.float64)
    
    # EMA0 initialization: first value is first data element
    if len(source_values) > 0:
        # Find first non-NaN value
        start_idx = 0
        for i, val in enumerate(source_values):
            if not np.isnan(val):
                start_idx = i
                break
        else:
            start_idx = len(source_values)
        
        if start_idx < len(source_values):
            # Initialize with first value
            ema_value = source_values[start_idx]
            expected_ema0[start_idx] = ema_value
            
            # Calculate EMA0 with alpha = 2.0 / (period + 1)
            alpha = 2.0 / (period + 1)
            alpha_n = 1.0 - alpha
            
            for i in range(start_idx + 1, len(source_values)):
                ema_value = source_values[i] * alpha + ema_value * alpha_n
                expected_ema0[i] = ema_value
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        expected_ema0
    ), f"MA (ma_type='ema0', period={period}) does not match direct calculation"


@pytest.mark.parametrize('period', [1, 2, 5, 8, 10, 22])
def test_ma_mma_direct_calculation(test_ohlcv_data, period):
    """Test MA with ma_type='mma' by direct calculation."""
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
    
    # Calculate with MA
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='mma')
    
    # Calculate expected MMA by direct computation
    # MMA uses iema_calculate: initialization via SMA of first period elements starting from first non-NaN
    source_values = close_data
    expected_mma = np.full(len(source_values), np.nan, dtype=np.float64)
    
    if len(source_values) >= period:
        # Find first non-NaN value
        start_idx = 0
        for i, val in enumerate(source_values):
            if not np.isnan(val):
                start_idx = i
                break
        
        # Calculate initial SMA (first period elements starting from start_idx)
        if start_idx + period <= len(source_values):
            initial_sma = source_values[start_idx:start_idx + period].sum() / period
            # First value is set at start_idx + period - 1
            expected_mma[start_idx + period - 1] = initial_sma
            
            # Calculate MMA with alpha = 1.0 / period
            alpha = 1.0 / period
            alpha_n = 1.0 - alpha
            mma_value = initial_sma
            
            for i in range(start_idx + period, len(source_values)):
                mma_value = source_values[i] * alpha + mma_value * alpha_n
                expected_mma[i] = mma_value
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        expected_mma
    ), f"MA (ma_type='mma', period={period}) does not match direct calculation"


@pytest.mark.parametrize('period', [2, 5, 8, 10, 22])
def test_ma_emaw_vs_talib(test_ohlcv_data, period):
    """Test MA with ma_type='emaw' against TA-Lib EMA.
    
    This test:
    1. Loads test OHLCV data
    2. Creates Quotes object
    3. Calculates EMA with warm-up using pyita (ma_type='emaw')
    4. Calculates EMA using TA-Lib (which uses warm-up method)
    5. Compares results
    
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
    ma_result = ta.ma(quotes, period=period, value='close', ma_type='emaw')
    
    # Calculate with TA-Lib
    talib_ema = talib.EMA(close_data, timeperiod=period)
    
    # Compare results
    assert arrays_equal_with_nan(
        ma_result.move_average,
        talib_ema
    ), f"MA (ma_type='emaw', period={period}) does not match TA-Lib EMA"

