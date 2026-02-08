"""Tests for ADX indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('period', [14, 2])
def test_adx_vs_si(test_ohlcv_data, period):
    """Test ADX calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    adx_result = ta.adx(quotes, period=period, smooth=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_adx', period)

    assert arrays_equal_with_nan(
        adx_result.adx, ref.adx
    ), f"ADX (period={period}) does not match stock-indicators"


@pytest.mark.parametrize('period', [14, 2])
def test_plus_di_vs_si(test_ohlcv_data, period):
    """Test Plus DI calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    adx_result = ta.adx(quotes, period=period, smooth=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_adx', period)

    assert arrays_equal_with_nan(
        adx_result.p_di[100:], ref.pdi[100:]
    ), f"Plus DI (period={period}) does not match stock-indicators"


@pytest.mark.parametrize('period', [14, 2])
def test_minus_di_vs_si(test_ohlcv_data, period):
    """Test Minus DI calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    adx_result = ta.adx(quotes, period=period, smooth=period)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_adx', period)

    assert arrays_equal_with_nan(
        adx_result.m_di[100:], ref.mdi[100:]
    ), f"Minus DI (period={period}) does not match stock-indicators"
