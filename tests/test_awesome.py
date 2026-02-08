"""Tests for Awesome Oscillator indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period_fast, period_slow', [
    (2, 5),
    (3, 7),
    (15, 20),
    (12, 20),
])
def test_awesome_vs_si(test_ohlcv_data, period_fast, period_slow):
    """Test Awesome Oscillator calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    awesome_result = ta.awesome(quotes, period_fast=period_fast, period_slow=period_slow, normalized=False)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_awesome', period_fast, period_slow)

    assert arrays_equal_with_nan(
        awesome_result.awesome, ref.oscillator
    ), f"Awesome Oscillator (period_fast={period_fast}, period_slow={period_slow}) does not match stock-indicators"
