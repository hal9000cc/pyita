"""williams_r(quotes, period=14)
Williams %R oscillator."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


@nb.njit(cache=True)
def calc_williams(high, low, close, period):
    """Calculate Williams %R oscillator.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Period for calculation
        
    Returns:
        Array of Williams %R values (first period-1 elements are NaN)
    """
    n_bars = len(high)

    williams_r = np.empty(n_bars, dtype=float)

    williams_r[: period - 1] = np.nan
    for t in range(period - 1, n_bars):
        high_max = high[t - period + 1: t + 1].max()
        low_min = low[t - period + 1: t + 1].min()
        williams_r[t] = 0 if high_max == low_min else (close[t] - high_max) / (high_max - low_min) * 100

    return williams_r


def get_indicator_out(quotes, period=14):
    """Calculate Williams %R oscillator.
    
    Williams %R is a momentum oscillator that measures overbought and oversold levels.
    It ranges from -100 to 0, with values above -20 typically indicating overbought
    conditions and values below -80 indicating oversold conditions.
    Formula: %R = (close - highest_high) / (highest_high - lowest_low) * 100
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for calculation (default: 14)
        
    Returns:
        IndicatorResult object with attribute:
            - williams_r: Williams %R values (-100 to 0, first period-1 elements are NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> williams_result = williams_r(quotes, period=14)
        >>> print(williams_result.williams_r)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    data_len = len(close)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    williams_r = calc_williams(high, low, close, period)
    
    return IndicatorResult({
        'williams_r': williams_r
    })

