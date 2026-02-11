"""supertrend(quotes, period=10, multipler=3, ma_type='mma')

Supertrend indicator.

Output series: supertrend (price), supertrend_mid (price)"""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..move_average import MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from . import atr


@nb.njit(cache=True)
def calc_supertrend(close, high, low, atr_values, multiplier, period):
    """Calculate Supertrend values.
    
    Args:
        close: Array of close prices
        high: Array of high prices
        low: Array of low prices
        atr_values: Array of ATR values
        multiplier: Multiplier for ATR
        period: Period for calculation
        
    Returns:
        Tuple of (supertrend, supertrend_mid) arrays
    """
    start_calculation = period - 1
    data_length = len(close)

    super_trend = np.empty(data_length, dtype=np.float64)
    super_trand_mid = np.empty(data_length, dtype=np.float64)
    super_trend[:start_calculation] = np.nan
    super_trand_mid[:start_calculation] = np.nan

    mid = (high[start_calculation] + low[start_calculation]) / 2.0
    upper_band = mid + (multiplier * atr_values[start_calculation])
    lower_band = mid - (multiplier * atr_values[start_calculation])
    trend_up = close[start_calculation] >= mid

    for i in range(start_calculation, len(close)):

        mid = (high[i] + low[i]) / 2.0
        super_trand_mid[i] = mid
        base_upper = mid + (multiplier * atr_values[i])
        base_lower = mid - (multiplier * atr_values[i])

        if base_upper < upper_band or close[i - 1] > upper_band:
            upper_band = base_upper

        if base_lower > lower_band or close[i - 1] < lower_band:
            lower_band = base_lower

        if close[i] <= (lower_band if trend_up else upper_band):
            super_trend[i] = upper_band
            trend_up = False
        else:
            super_trend[i] = lower_band
            trend_up = True

    return super_trend, super_trand_mid


def get_indicator_out(quotes, period=10, multipler=3, ma_type='mma'):
    """Calculate Supertrend indicator.
    
    Supertrend is a trend-following indicator that uses ATR to determine trend direction.
    It provides buy and sell signals based on price crossing above or below the Supertrend line.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for ATR calculation (default: 10)
        multipler: Multiplier for ATR (default: 3)
        ma_type: Type of moving average for ATR - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'mma')
        
    Returns:
        IndicatorResult object with attributes:
            - supertrend: Supertrend line values (first period-1 elements are NaN)
            - supertrend_mid: Mid line values (first period-1 elements are NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, multipler <= 0, or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> supertrend_result = supertrend(quotes, period=10, multipler=3)
        >>> print(supertrend_result.supertrend)
        >>> print(supertrend_result.supertrend_mid)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    if multipler <= 0:
        raise PyTAExceptionBadParameterValue(f'multipler must be greater than 0, got {multipler}')
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    data_len = len(close)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    atr_result = atr.get_indicator_out(quotes, smooth=period, ma_type=ma_type)
    atr_values = atr_result.atr
    
    supertrend, supertrend_mid = calc_supertrend(close, high, low, atr_values, multipler, period)
    
    return IndicatorResult({
        'supertrend': supertrend,
        'supertrend_mid': supertrend_mid
    })

