"""Tests for TEMA indicator."""
import pytest
import py_ta as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [
    1,
    5,
    22,
    14,
])
def test_tema_vs_si(test_ohlcv_data, period):
    """Test TEMA calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    tema_result = ta.tema(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_tema', period)

    assert arrays_equal_with_nan(
        tema_result.tema, ref.tema
    ), f"TEMA (period={period}) does not match stock-indicators"

