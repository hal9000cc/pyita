"""Tests for Ichimoku indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period_short, period_mid, period_long, offset_senkou, offset_chikou', [
    (9, 26, 52, 26, 26),
    (9, 26, 52, 25, 27),
])
def test_ichimoku_vs_si(test_ohlcv_data, period_short, period_mid, period_long, offset_senkou, offset_chikou):
    """Test Ichimoku calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    ichimoku_result = ta.ichimoku(
        quotes,
        period_short=period_short,
        period_mid=period_mid,
        period_long=period_long,
        offset_senkou=offset_senkou,
        offset_chikou=offset_chikou
    )

    ref = get_si_ref(
        TEST_DATA_FILENAME,
        'get_ichimoku',
        period_short,
        period_mid,
        period_long,
        offset_senkou,
        offset_chikou
    )

    assert arrays_equal_with_nan(
        ichimoku_result.tenkan, ref.tenkan_sen
    ), f"Ichimoku Tenkan (period_short={period_short}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        ichimoku_result.kijun, ref.kijun_sen
    ), f"Ichimoku Kijun (period_mid={period_mid}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        ichimoku_result.senkou_a, ref.senkou_span_a
    ), f"Ichimoku Senkou A (offset_senkou={offset_senkou}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        ichimoku_result.senkou_b, ref.senkou_span_b
    ), f"Ichimoku Senkou B (period_long={period_long}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        ichimoku_result.chikou, ref.chikou_span
    ), f"Ichimoku Chikou (offset_chikou={offset_chikou}) does not match stock-indicators"
