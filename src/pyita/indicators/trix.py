"""trix(quotes, period, value='close')
Triple Exponential Average Oscillator."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue


def get_indicator_out(quotes, period, value='close'):
    """Calculate Triple Exponential Average Oscillator (TRIX).
    
    TRIX is a momentum oscillator that shows the rate of change of a triple
    exponentially smoothed moving average. It filters out market noise and
    highlights significant price movements.
    Formula: TRIX = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
    where EMA1 = EMA(source), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attribute:
            - trix: Triple exponential average oscillator values (first element is NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or value is not a valid price field
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is less than required
        
    Example:
        >>> trix_result = trix(quotes, period=14, value='close')
        >>> print(trix_result.trix)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    source_values = quotes[value]
    
    ema1 = ma_calculate(source_values, period, MA_Type.ema)
    ema2 = ma_calculate(ema1, period, MA_Type.ema0)
    ema3 = ma_calculate(ema2, period, MA_Type.ema0)
    
    np.seterr(divide='ignore', invalid='ignore')
    trix = np.diff(ema3) / ema3[:-1] * 100
    
    return IndicatorResult({
        'trix': np.hstack([np.nan, trix])
    })

