"""Tests for ATR indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('smooth', [2, 14])
def test_atr_vs_si(test_ohlcv_data, smooth):
    """Test ATR calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    atr_result = ta.atr(quotes, smooth=smooth, ma_type='mma')

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_atr', smooth)

    assert arrays_equal_with_nan(
        atr_result.tr[1:], ref.tr[1:]
    ), f"TR (smooth={smooth}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        atr_result.atr[200:], ref.atr[200:]
    ), f"ATR (smooth={smooth}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        atr_result.atrp[200:], ref.atrp[200:]
    ), f"ATRP (smooth={smooth}) does not match stock-indicators"
