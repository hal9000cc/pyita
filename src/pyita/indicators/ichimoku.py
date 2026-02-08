"""ichimoku(quotes, period_short=9, period_mid=26, period_long=52, offset_senkou=26, offset_chikou=26)
Ichimoku indicator."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from ..constants import PRICE_TYPE


@nb.njit(cache=True)
def calc_av_min_max(high, low, period):
    """Calculate average of maximum and minimum over a period.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        period: Period for calculation
        
    Returns:
        Array of average (max + min) / 2 values
    """
    n_bars = len(high)

    av_min_max = np.empty(n_bars, dtype=np.float64)

    av_min_max[:period - 1] = np.nan
    for t in range(period - 1, n_bars):
        av_min_max[t] = (high[t - period + 1: t + 1].max() + low[t - period + 1: t + 1].min()) / 2

    return av_min_max


def offset_ahead(series, period):
    """Shift series forward (to the right) by period positions, filling beginning with NaN.
    
    This function modifies the array in-place.
    
    Args:
        series: Array to shift
        period: Number of positions to shift forward
    """
    if period > 0:
        series[period:] = series[:-period]
        series[:period] = np.nan


def get_indicator_out(quotes, period_short=9, period_mid=26, period_long=52, offset_senkou=26, offset_chikou=26):
    """Calculate Ichimoku Cloud indicator.
    
    Ichimoku is a comprehensive technical analysis system that provides support and
    resistance levels, trend direction, and momentum. It consists of five lines:
    - Tenkan-sen (Conversion Line): average of highest high and lowest low over period_short
    - Kijun-sen (Base Line): average of highest high and lowest low over period_mid
    - Senkou Span A (Leading Span A): average of Tenkan and Kijun, shifted forward
    - Senkou Span B (Leading Span B): average of highest high and lowest low over period_long, shifted forward
    - Chikou Span (Lagging Span): close price, shifted backward
    
    Args:
        quotes: Quotes object containing OHLCV data
        period_short: Period for Tenkan-sen calculation (default: 9)
        period_mid: Period for Kijun-sen calculation (default: 26)
        period_long: Period for Senkou Span B calculation (default: 52)
        offset_senkou: Offset for shifting Senkou spans forward (default: 26)
        offset_chikou: Offset for shifting Chikou span backward (default: 26)
        
    Returns:
        IndicatorResult object with attributes:
            - tenkan: Tenkan-sen (Conversion Line) values
            - kijun: Kijun-sen (Base Line) values
            - senkou_a: Senkou Span A (Leading Span A) values
            - senkou_b: Senkou Span B (Leading Span B) values
            - chikou: Chikou Span (Lagging Span) values
            
    Raises:
        PyTAExceptionBadParameterValue: If any period <= 0 or offset < 0
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> ichimoku_result = ichimoku(quotes, period_short=9, period_mid=26, period_long=52)
        >>> print(ichimoku_result.tenkan)
        >>> print(ichimoku_result.kijun)
    """
    # Validate periods
    if period_short <= 0:
        raise PyTAExceptionBadParameterValue(f'period_short must be greater than 0, got {period_short}')
    if period_mid <= 0:
        raise PyTAExceptionBadParameterValue(f'period_mid must be greater than 0, got {period_mid}')
    if period_long <= 0:
        raise PyTAExceptionBadParameterValue(f'period_long must be greater than 0, got {period_long}')
    
    # Validate offsets
    if offset_senkou < 0:
        raise PyTAExceptionBadParameterValue(f'offset_senkou must be >= 0, got {offset_senkou}')
    if offset_chikou < 0:
        raise PyTAExceptionBadParameterValue(f'offset_chikou must be >= 0, got {offset_chikou}')
    
    # Get OHLC data from quotes
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    # Check minimum data requirement
    max_period = max(period_short, period_mid, period_long)
    data_len = len(high)
    if data_len < max_period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {max_period}')
    
    # Calculate Tenkan-sen and Kijun-sen
    tenkan = calc_av_min_max(high, low, period_short)
    kijun = calc_av_min_max(high, low, period_mid)
    
    # Calculate Senkou Span A (average of Tenkan and Kijun, shifted forward)
    senkou_a = np.vstack((tenkan, kijun)).sum(0) / 2
    # Make a copy before shifting to avoid modifying the original
    senkou_a = senkou_a.copy()
    offset_ahead(senkou_a, offset_senkou)
    
    # Calculate Senkou Span B (average max/min over period_long, shifted forward)
    senkou_b = calc_av_min_max(high, low, period_long)
    # Make a copy before shifting
    senkou_b = senkou_b.copy()
    offset_ahead(senkou_b, offset_senkou)
    
    # Calculate Chikou Span (close shifted backward)
    chikou = np.empty(len(high), dtype=PRICE_TYPE)
    if offset_chikou > 0:
        chikou[:-offset_chikou] = close[offset_chikou:]
        chikou[-offset_chikou:] = np.nan
    else:
        chikou[:] = close
    
    return IndicatorResult({
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou
    })

