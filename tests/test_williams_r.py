"""Tests for Williams %R indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [
    2,
    1,
    5,
    22,
    22,
])
def test_williams_r_vs_si(test_ohlcv_data, period):
    """Test Williams %R calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    williams_r_result = ta.williams_r(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_williams_r', period)

    assert arrays_equal_with_nan(
        williams_r_result.williams_r, ref.williams_r
    ), f"Williams %R (period={period}) does not match stock-indicators"

