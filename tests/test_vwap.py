"""Tests for VWAP indicator."""
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


def test_vwap_vs_si(test_ohlcv_data):
    """Test VWAP calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    vwap_result = ta.vwap(quotes)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_vwap')

    assert arrays_equal_with_nan(
        vwap_result.vwap, ref.vwap
    ), "VWAP does not match stock-indicators"

