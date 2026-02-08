"""Tests for Keltner Channel indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period, multiplier, period_atr', [
    (2, 1, 2),
    (5, 2, 5),
    (10, 3, 7),
])
def test_keltner_vs_si(test_ohlcv_data, period, multiplier, period_atr):
    """Test Keltner Channel calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    keltner_result = ta.keltner(
        quotes,
        period=period,
        multiplier=multiplier,
        period_atr=period_atr,
        ma_type='ema',
        ma_type_atr='mma'
    )

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_keltner', period, multiplier, period_atr)

    assert arrays_equal_with_nan(
        keltner_result.mid_line, ref.center_line
    ), f"Keltner Mid Line (period={period}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        keltner_result.up_line[100:], ref.upper_band[100:]
    ), f"Keltner Upper Line (period={period}, multiplier={multiplier}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        keltner_result.down_line[100:], ref.lower_band[100:]
    ), f"Keltner Lower Line (period={period}, multiplier={multiplier}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        keltner_result.width[100:], ref.width[100:]
    ), f"Keltner Width (period={period}, multiplier={multiplier}) does not match stock-indicators"
