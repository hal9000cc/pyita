"""Tests for TRIX indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [
    1,
    5,
    22,
    14,
])
def test_trix_vs_si(test_ohlcv_data, period):
    """Test TRIX calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    trix_result = ta.trix(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_trix', period)

    assert arrays_equal_with_nan(
        trix_result.trix, ref.trix
    ), f"TRIX (period={period}) does not match stock-indicators"

