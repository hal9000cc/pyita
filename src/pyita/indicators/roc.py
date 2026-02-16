"""roc(quotes, period=14, ma_period=14, ma_type='sma', value='close')

Rate of Change.

Output series: roc, smooth_roc"""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from ..constants import PRICE_TYPE
from ..helpers import validate_value_par


def get_indicator_out(quotes, period=14, ma_period=14, ma_type='sma', value='close'):
    """Calculate Rate of Change (ROC).
    
    ROC is a momentum oscillator that measures the percentage change in price
    over a specified period. It shows the speed at which price is changing.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for ROC calculation (default: 14)
        ma_period: Period for smoothing ROC (default: 14)
        ma_type: Type of moving average for smoothing - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attributes:
            - roc: Rate of Change values (first period elements are NaN)
            - smooth_roc: Smoothed ROC values (first period elements are NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, ma_period <= 0, value is invalid, or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> roc_result = roc(quotes, period=14, ma_period=14)
        >>> print(roc_result.roc)
        >>> print(roc_result.smooth_roc)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    if ma_period <= 0:
        raise PyTAExceptionBadParameterValue(f'ma_period must be greater than 0, got {ma_period}')
    
    validate_value_par(value, allow_volume=True)
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    source_values = quotes[value]
    
    data_len = len(source_values)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    roc = (source_values[period:] - source_values[:-period]) / source_values[:-period] * 100
    
    np.seterr(divide='ignore', invalid='ignore')
    roc[source_values[:-period] == 0] = 0
    
    smooth_roc = ma_calculate(roc, ma_period, ma_type_enum)
    
    begin = np.array([np.nan] * period, dtype=PRICE_TYPE)
    
    return IndicatorResult({
        'roc': np.hstack((begin, roc)),
        'smooth_roc': np.hstack((begin, smooth_roc))
    })

