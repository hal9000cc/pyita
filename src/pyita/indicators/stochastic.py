"""stochastic(quotes, period=5, period_d=3, smooth=3, ma_type='sma')
Stochastic oscillator."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


@nb.njit(cache=True)
def calc_k(high, low, close, period):
    """Calculate %K (raw stochastic oscillator).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Period for calculation
        
    Returns:
        Array of %K values (0-100, first period-1 elements are NaN)
    """
    value_k = np.empty(len(close), dtype=np.float64)
    value_k[:period - 1] = np.nan

    for i in range(period - 1, len(close)):
        v_high = high[i - period + 1: i + 1].max()
        v_low = low[i - period + 1: i + 1].min()
        value_k[i] = 0 if v_high == v_low else (close[i] - v_low) / (v_high - v_low) * 100

    return value_k


def get_indicator_out(quotes, period=5, period_d=3, smooth=3, ma_type='sma'):
    """Calculate Stochastic Oscillator.
    
    Stochastic Oscillator is a momentum indicator that compares a closing price
    to its price range over a given period. It consists of:
    - %K (oscillator): Raw stochastic value
    - %K (value_k): Smoothed %K
    - %D (value_d): Moving average of %K
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for %K calculation (default: 5)
        period_d: Period for %D calculation (default: 3)
        smooth: Period for smoothing %K (default: 3)
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        
    Returns:
        IndicatorResult object with attributes:
            - oscillator: Raw %K values (0-100, first period-1 elements are NaN)
            - value_k: Smoothed %K values
            - value_d: %D values (moving average of %K)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, period_d <= 0, smooth <= 0, or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> stoch_result = stochastic(quotes, period=5, period_d=3, smooth=3)
        >>> print(stoch_result.oscillator)
        >>> print(stoch_result.value_k)
        >>> print(stoch_result.value_d)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    if period_d <= 0:
        raise PyTAExceptionBadParameterValue(f'period_d must be greater than 0, got {period_d}')
    if smooth <= 0:
        raise PyTAExceptionBadParameterValue(f'smooth must be greater than 0, got {smooth}')
    
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
    
    oscillator = calc_k(high, low, close, period)
    
    value_k = ma_calculate(oscillator, smooth, ma_type_enum)
    
    value_d = ma_calculate(value_k, period_d, ma_type_enum)
    
    return IndicatorResult({
        'oscillator': oscillator,
        'value_k': value_k,
        'value_d': value_d
    })

