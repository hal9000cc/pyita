"""chandelier(quotes, period=22, multiplier=3, use_close=False)
Chandelier Exit."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from . import atr


@nb.njit(cache=True)
def calc_chandelier(high, low, atr_values, period, multiplier):
    """Calculate Chandelier Exit values.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        atr_values: Array of ATR values
        period: Period for finding extremes
        multiplier: Multiplier for ATR
        
    Returns:
        Tuple of (exit_short, exit_long) arrays
    """
    n_bars = len(high)
    exit_short = np.empty(n_bars, dtype=np.float64)
    exit_long = np.empty(n_bars, dtype=np.float64)

    exit_long[:period - 1] = np.nan
    exit_short[:period - 1] = np.nan
    for i in range(period, n_bars + 1):
        exit_long[i - 1] = high[i - period: i].max() - atr_values[i - 1] * multiplier
        exit_short[i - 1] = low[i - period: i].min() + atr_values[i - 1] * multiplier

    return exit_short, exit_long


def get_indicator_out(quotes, period=22, multiplier=3, use_close=False):
    """Calculate Chandelier Exit.
    
    Chandelier Exit is a volatility-based indicator that uses ATR to set trailing
    stop-loss levels. It calculates exit points for long and short positions based
    on the highest high (or close) and lowest low (or close) over a period, adjusted
    by ATR multiplied by a factor.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period used for ATR and for finding extremes (default: 22)
        multiplier: Multiplier for ATR (default: 3)
        use_close: If True, close is used to calculate the values, otherwise high and low are used (default: False)
        
    Returns:
        IndicatorResult object with attributes:
            - exit_long: Exit level for long positions
            - exit_short: Exit level for short positions
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or multiplier <= 0
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> chandelier_result = chandelier(quotes, period=22, multiplier=3)
        >>> print(chandelier_result.exit_long)
        >>> print(chandelier_result.exit_short)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Validate multiplier
    if multiplier <= 0:
        raise PyTAExceptionBadParameterValue(f'multiplier must be greater than 0, got {multiplier}')
    
    # Get OHLC data from quotes
    close = quotes.close
    
    # Check minimum data requirement
    data_len = len(close)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    # Calculate ATR
    atr_result = atr.get_indicator_out(quotes, smooth=period, ma_type='mma')
    atr_values = atr_result.atr
    
    # Determine high and low based on use_close parameter
    if use_close:
        high = close
        low = close
    else:
        high = quotes.high
        low = quotes.low
    
    # Calculate Chandelier Exit
    exit_short, exit_long = calc_chandelier(high, low, atr_values, period, multiplier)
    
    return IndicatorResult({
        'exit_short': exit_short,
        'exit_long': exit_long
    })

