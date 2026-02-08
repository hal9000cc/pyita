"""Tests for Stochastic Oscillator indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period, period_d, smooth', [
    (1, 5, 1),
    (1, 5, 3),
    (1, 1, 1),
    (2, 5, 3),
    (14, 5, 3),
])
def test_stochastic_vs_si(test_ohlcv_data, period, period_d, smooth):
    """Test Stochastic Oscillator calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    stoch_result = ta.stochastic(quotes, period=period, period_d=period_d, smooth=smooth, ma_type='sma')

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_stoch', period, period_d, smooth)

    if smooth == 1:
        assert arrays_equal_with_nan(
            stoch_result.oscillator, ref.oscillator
        ), f"Stochastic oscillator (period={period}, smooth={smooth}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        stoch_result.value_k, ref.k
    ), f"Stochastic %K (period={period}, smooth={smooth}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        stoch_result.value_d, ref.d
    ), f"Stochastic %D (period_d={period_d}) does not match stock-indicators"
