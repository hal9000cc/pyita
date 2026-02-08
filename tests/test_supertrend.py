"""Tests for Supertrend indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [2, 20])
def test_supertrend_vs_si(test_ohlcv_data, period):
    """Test Supertrend calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    supertrend_result = ta.supertrend(quotes, period=period, multipler=3, ma_type='mma')

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_super_trend', period, 3)

    assert arrays_equal_with_nan(
        supertrend_result.supertrend[200:], ref.super_trend[200:]
    ), f"Supertrend (period={period}) does not match stock-indicators"
