"""Tests for Quotes class."""
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyita as ta
from pyita.exceptions import PyTAExceptionBadSeriesData


# Test data filenames
TEST_DATA_1H_FILENAME = "BINANCE_BTC_USDT_1h_2025.pkl"
TEST_DATA_1D_FILENAME = "BINANCE_BTC_USDT_1d_2025.pkl"


@pytest.fixture
def test_data_100():
    """Load and return first 100 elements from hourly test data.
    
    Returns:
        dict: Dictionary with keys 'time', 'open', 'high', 'low', 'close', 'volume'
            containing numpy arrays of OHLCV data (100 elements)
    """
    test_data_dir = Path(__file__).parent / "test_data"
    filepath = test_data_dir / TEST_DATA_1H_FILENAME
    
    assert filepath.exists(), f"Test data file not found: {filepath}"
    
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Trim to 100 elements
    trimmed_data = {
        'time': data_dict['time'][:100],
        'open': data_dict['open'][:100],
        'high': data_dict['high'][:100],
        'low': data_dict['low'][:100],
        'close': data_dict['close'][:100],
        'volume': data_dict['volume'][:100],
    }
    
    return trimmed_data


@pytest.fixture
def test_data_1d_100():
    """Load and return first 100 elements from daily test data.
    
    Returns:
        dict: Dictionary with keys 'time', 'open', 'high', 'low', 'close', 'volume'
            containing numpy arrays of OHLCV data (100 elements)
    """
    test_data_dir = Path(__file__).parent / "test_data"
    filepath = test_data_dir / TEST_DATA_1D_FILENAME
    
    assert filepath.exists(), f"Test data file not found: {filepath}"
    
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Trim to 100 elements
    trimmed_data = {
        'time': data_dict['time'][:100],
        'open': data_dict['open'][:100],
        'high': data_dict['high'][:100],
        'low': data_dict['low'][:100],
        'close': data_dict['close'][:100],
        'volume': data_dict['volume'][:100],
    }
    
    return trimmed_data


def assert_quotes_equal(quotes, expected_data):
    """Assert that Quotes object matches expected data.
    
    Checks:
    - Data types are correct (float for prices, datetime64[ms] for time)
    - Values match expected data
    
    Args:
        quotes: Quotes object to check
        expected_data: Dictionary with expected values
            Keys: 'open', 'high', 'low', 'close', 'volume' (optional), 'time' (optional)
    """
    # Check price data types (should be float)
    assert quotes.open.dtype == np.float64, "open should be float64"
    assert quotes.high.dtype == np.float64, "high should be float64"
    assert quotes.low.dtype == np.float64, "low should be float64"
    assert quotes.close.dtype == np.float64, "close should be float64"
    
    # Check values
    np.testing.assert_array_almost_equal(quotes.open, expected_data['open'], decimal=10)
    np.testing.assert_array_almost_equal(quotes.high, expected_data['high'], decimal=10)
    np.testing.assert_array_almost_equal(quotes.low, expected_data['low'], decimal=10)
    np.testing.assert_array_almost_equal(quotes.close, expected_data['close'], decimal=10)
    
    # Check volume if provided
    if 'volume' in expected_data:
        assert quotes.volume.dtype == np.float64, "volume should be float64"
        np.testing.assert_array_almost_equal(quotes.volume, expected_data['volume'], decimal=10)
    else:
        assert not hasattr(quotes, 'volume') or quotes.volume is None, "volume should not be present"
    
    # Check time if provided
    if 'time' in expected_data:
        assert quotes.time.dtype == np.dtype('datetime64[ms]'), "time should be datetime64[ms]"
        np.testing.assert_array_equal(quotes.time, expected_data['time'])
    else:
        assert not hasattr(quotes, 'time') or quotes.time is None, "time should not be present"


def test_quotes_from_4_arrays(test_data_100):
    """Test Quotes creation from 4 numpy arrays (open, high, low, close)."""
    quotes = ta.Quotes(
        test_data_100['open'],
        test_data_100['high'],
        test_data_100['low'],
        test_data_100['close']
    )
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_5_arrays(test_data_100):
    """Test Quotes creation from 5 numpy arrays (+ volume)."""
    quotes = ta.Quotes(
        test_data_100['open'],
        test_data_100['high'],
        test_data_100['low'],
        test_data_100['close'],
        test_data_100['volume']
    )
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_6_arrays(test_data_100):
    """Test Quotes creation from 6 numpy arrays (+ time)."""
    quotes = ta.Quotes(
        test_data_100['open'],
        test_data_100['high'],
        test_data_100['low'],
        test_data_100['close'],
        test_data_100['volume'],
        test_data_100['time']
    )
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
        'time': test_data_100['time'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_kwargs(test_data_100):
    """Test Quotes creation from kwargs dictionary."""
    quotes = ta.Quotes(**{
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
        'time': test_data_100['time'],
    })
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
        'time': test_data_100['time'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_int_arrays_datetime64us(test_data_100):
    """Test Quotes creation from int numpy arrays and datetime64[us]."""
    # Convert to int arrays
    open_int = test_data_100['open'].astype(int)
    high_int = test_data_100['high'].astype(int)
    low_int = test_data_100['low'].astype(int)
    close_int = test_data_100['close'].astype(int)
    volume_int = test_data_100['volume'].astype(int)
    
    # Convert time to datetime64[us]
    time_us = test_data_100['time'].astype('datetime64[us]')
    
    quotes = ta.Quotes(open_int, high_int, low_int, close_int, volume_int, time_us)
    
    # Expected data: int -> float conversion (same as what Quotes does)
    expected = {
        'open': test_data_100['open'].astype(int).astype(float),
        'high': test_data_100['high'].astype(int).astype(float),
        'low': test_data_100['low'].astype(int).astype(float),
        'close': test_data_100['close'].astype(int).astype(float),
        'volume': test_data_100['volume'].astype(int).astype(float),
        'time': test_data_100['time'],  # Should be converted to datetime64[ms]
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_float_lists_datetime(test_data_100):
    """Test Quotes creation from float lists and Python datetime."""
    # Convert to lists
    open_list = test_data_100['open'].tolist()
    high_list = test_data_100['high'].tolist()
    low_list = test_data_100['low'].tolist()
    close_list = test_data_100['close'].tolist()
    volume_list = test_data_100['volume'].tolist()
    
    # Convert time to Python datetime
    time_datetime = [pd.Timestamp(ts).to_pydatetime() for ts in test_data_100['time']]
    
    quotes = ta.Quotes(open_list, high_list, low_list, close_list, volume_list, time_datetime)
    
    # Expected data should be original numpy arrays
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
        'time': test_data_100['time'],  # Should be converted to datetime64[ms]
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_int_lists_date(test_data_1d_100):
    """Test Quotes creation from int lists and Python date."""
    # Convert to int lists
    open_list = [int(x) for x in test_data_1d_100['open']]
    high_list = [int(x) for x in test_data_1d_100['high']]
    low_list = [int(x) for x in test_data_1d_100['low']]
    close_list = [int(x) for x in test_data_1d_100['close']]
    volume_list = [int(x) for x in test_data_1d_100['volume']]
    
    # Convert time to Python date (start of day)
    time_date = [pd.Timestamp(ts).date() for ts in test_data_1d_100['time']]
    
    quotes = ta.Quotes(open_list, high_list, low_list, close_list, volume_list, time_date)
    
    # Expected time should be start of day (00:00:00) in datetime64[ms]
    expected_time = np.array([np.datetime64(d, 'ms') for d in time_date])
    
    # Expected data: int -> float conversion (same as what Quotes does)
    expected = {
        'open': test_data_1d_100['open'].astype(int).astype(float),
        'high': test_data_1d_100['high'].astype(int).astype(float),
        'low': test_data_1d_100['low'].astype(int).astype(float),
        'close': test_data_1d_100['close'].astype(int).astype(float),
        'volume': test_data_1d_100['volume'].astype(int).astype(float),
        'time': expected_time,
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_pandas_4_columns(test_data_100):
    """Test Quotes creation from pandas DataFrame with 4 columns (mixed case)."""
    df = pd.DataFrame({
        'open': test_data_100['open'],
        'High': test_data_100['high'],
        'low': test_data_100['low'],
        'Close': test_data_100['close'],
    })
    
    quotes = ta.Quotes(df)
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_pandas_5_columns(test_data_100):
    """Test Quotes creation from pandas DataFrame with 5 columns (mixed case)."""
    df = pd.DataFrame({
        'Open': test_data_100['open'],
        'high': test_data_100['high'],
        'Low': test_data_100['low'],
        'close': test_data_100['close'],
        'Volume': test_data_100['volume'],
    })
    
    quotes = ta.Quotes(df)
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_from_pandas_6_columns(test_data_100):
    """Test Quotes creation from pandas DataFrame with 6 columns (mixed case)."""
    df = pd.DataFrame({
        'open': test_data_100['open'],
        'HIGH': test_data_100['high'],
        'low': test_data_100['low'],
        'CLOSE': test_data_100['close'],
        'volume': test_data_100['volume'],
        'Time': test_data_100['time'],
    })
    
    quotes = ta.Quotes(df)
    
    expected = {
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        'close': test_data_100['close'],
        'volume': test_data_100['volume'],
        'time': test_data_100['time'],
    }
    
    assert_quotes_equal(quotes, expected)


def test_quotes_validation_different_lengths(test_data_100):
    """Test Quotes validation with arrays of different lengths."""
    # Create arrays with different lengths
    open_data = test_data_100['open'][:50]
    high_data = test_data_100['high'][:100]
    low_data = test_data_100['low'][:75]
    close_data = test_data_100['close'][:100]
    
    with pytest.raises(PyTAExceptionBadSeriesData):
        ta.Quotes(open_data, high_data, low_data, close_data)


def test_quotes_validation_missing_columns(test_data_100):
    """Test Quotes validation with missing required columns in pandas DataFrame."""
    # Missing 'close' column
    df = pd.DataFrame({
        'open': test_data_100['open'],
        'high': test_data_100['high'],
        'low': test_data_100['low'],
        # 'close' is missing
    })
    
    with pytest.raises(PyTAExceptionBadSeriesData):
        ta.Quotes(df)

