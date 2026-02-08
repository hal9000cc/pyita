"""Tests for Aroon indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [2, 7, 14])
def test_aroon_vs_si(test_ohlcv_data, period):
    """Test Aroon calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    aroon_result = ta.aroon(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_aroon', period)

    assert arrays_equal_with_nan(
        aroon_result.up, ref.aroon_up
    ), f"Aroon Up (period={period}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        aroon_result.down, ref.aroon_down
    ), f"Aroon Down (period={period}) does not match stock-indicators"


@pytest.mark.parametrize('period', [2, 7, 14])
def test_aroon_oscillator_vs_si(test_ohlcv_data, period):
    """Test Aroon Oscillator calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    aroon_result = ta.aroon(quotes, period=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_aroon', period)

    assert arrays_equal_with_nan(
        aroon_result.oscillator, ref.oscillator
    ), f"Aroon Oscillator (period={period}) does not match stock-indicators"
