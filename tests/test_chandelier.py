"""Tests for Chandelier Exit indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period, multiplier', [
    (2, 3),
    (20, 2.5),
])
def test_chandelier_vs_si(test_ohlcv_data, period, multiplier):
    """Test Chandelier Exit calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    chandelier_result = ta.chandelier(quotes, period=period, multiplier=multiplier, use_close=False)

    # Pass strings instead of enum - conversion happens inside get_si_ref only when generating data
    ref_short = get_si_ref(TEST_DATA_FILENAME, 'get_chandelier', period, multiplier, 'SHORT')
    ref_long = get_si_ref(TEST_DATA_FILENAME, 'get_chandelier', period, multiplier, 'LONG')

    assert arrays_equal_with_nan(
        chandelier_result.exit_short[200:], ref_short.chandelier_exit[200:]
    ), f"Chandelier Exit Short (period={period}, multiplier={multiplier}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        chandelier_result.exit_long[200:], ref_long.chandelier_exit[200:]
    ), f"Chandelier Exit Long (period={period}, multiplier={multiplier}) does not match stock-indicators"
