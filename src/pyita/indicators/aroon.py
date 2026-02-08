"""aroon(quotes, period=14)
Aroon oscillator."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


@nb.njit(cache=True)
def calc_aroon(high, low, period):
    """Calculate Aroon indicator values.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        period: Period for Aroon calculation
        
    Returns:
        Tuple of (up, down, oscillator) arrays
    """
    up = np.empty(len(high), dtype=np.float64)
    down = np.empty(len(high), dtype=np.float64)
    oscillator = np.empty(len(high), dtype=np.float64)

    up[:period] = np.nan
    down[:period] = np.nan
    oscillator[:period] = np.nan

    for i in range(period, len(high)):
        i_max = period - high[i - period: i + 1].argmax()
        i_min = period - low[i - period: i + 1].argmin()
        up[i] = (period - i_max) / period * 100
        down[i] = (period - i_min) / period * 100
        oscillator[i] = up[i] - down[i]

    return up, down, oscillator


def get_indicator_out(quotes, period=14):
    """Calculate Aroon oscillator.
    
    Aroon is a technical indicator used to identify trend changes and the strength
    of trends. It consists of two lines: Aroon Up and Aroon Down, which measure
    the time since the highest high and lowest low within a given period.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for Aroon calculation (default: 14)
        
    Returns:
        IndicatorResult object with attributes:
            - up: Aroon Up values (0-100)
            - down: Aroon Down values (0-100)
            - oscillator: Aroon Oscillator values (up - down, range -100 to 100)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> aroon_result = aroon(quotes, period=14)
        >>> print(aroon_result.up)
        >>> print(aroon_result.down)
        >>> print(aroon_result.oscillator)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    high = quotes.high
    low = quotes.low
    
    data_len = len(high)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    up, down, oscillator = calc_aroon(high, low, period)
    
    return IndicatorResult({
        'up': up,
        'down': down,
        'oscillator': oscillator
    })

