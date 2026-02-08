"""Tests for VWMA indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [
    2,
    14,
    15,
])
def test_vwma_vs_si(test_ohlcv_data, period):
    """Test VWMA calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    vwma_result = ta.vwma(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_vwma', period)

    assert arrays_equal_with_nan(
        vwma_result.vwma, ref.vwma
    ), f"VWMA (period={period}) does not match stock-indicators"

