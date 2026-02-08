"""Tests for Parabolic SAR indicator."""
import numpy as np
import pytest
import pyita as ta

from conftest import TEST_DATA_FILENAME, arrays_equal_with_nan
from stock_indicators_helpers import get_si_ref


@pytest.mark.parametrize('start, maximum, increment', [
    (0.02, 0.2, 0.02),
    (0.01, 0.2, 0.02),
    (0.02, 0.3, 0.01),
])
def test_parabolic_sar_vs_si(test_ohlcv_data, start, maximum, increment):
    """Test Parabolic SAR calculation against stock-indicators reference."""
    quotes = ta.Quotes(
        test_ohlcv_data['open'],
        test_ohlcv_data['high'],
        test_ohlcv_data['low'],
        test_ohlcv_data['close'],
        test_ohlcv_data['volume'],
        test_ohlcv_data['time'],
    )

    sar_result = ta.parabolic_sar(quotes, start=start, maximum=maximum, increment=increment)

    ref = get_si_ref(TEST_DATA_FILENAME, 'get_parabolic_sar', increment, maximum, start)

    ref_is_reversal = ref.is_reversal.copy()
    ref_is_reversal[np.isnan(ref_is_reversal)] = 0

    assert (np.abs(sar_result.signal) == ref_is_reversal).all(), \
        f"Parabolic SAR signal (start={start}, maximum={maximum}, increment={increment}) does not match stock-indicators"

    assert arrays_equal_with_nan(
        sar_result.sar, ref.sar
    ), f"Parabolic SAR (start={start}, maximum={maximum}, increment={increment}) does not match stock-indicators"
