"""awesome(quotes, period_fast=5, period_slow=34, ma_type_fast='sma', ma_type_slow='sma', normalized=False)
Awesome oscillator."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


def get_indicator_out(quotes, period_fast=5, period_slow=34, ma_type_fast='sma', ma_type_slow='sma', normalized=False):
    """Calculate Awesome Oscillator.
    
    Awesome Oscillator is a momentum indicator that measures the difference between
    a fast and slow moving average of the median price (high + low) / 2.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period_fast: Period for fast moving average (default: 5)
        period_slow: Period for slow moving average (default: 34)
        ma_type_fast: Type of moving average for fast MA - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        ma_type_slow: Type of moving average for slow MA - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        normalized: If True, normalize awesome by median price (default: False)
        
    Returns:
        IndicatorResult object with attribute:
            - awesome: Awesome Oscillator values
            
    Raises:
        PyTAExceptionBadParameterValue: If period_fast <= 0, period_slow <= 0, period_slow <= period_fast, or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is less than period_slow
        
    Example:
        >>> awesome_result = awesome(quotes, period_fast=5, period_slow=34)
        >>> print(awesome_result.awesome)
    """
    if period_fast <= 0:
        raise PyTAExceptionBadParameterValue(f'period_fast must be greater than 0, got {period_fast}')
    
    if period_slow <= 0:
        raise PyTAExceptionBadParameterValue(f'period_slow must be greater than 0, got {period_slow}')
    
    if period_slow <= period_fast:
        raise PyTAExceptionBadParameterValue(f'period_slow ({period_slow}) must be greater than period_fast ({period_fast})')
    
    try:
        ma_type_fast_enum = MA_Type.cast(ma_type_fast)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type_fast: {str(e)}')
    
    try:
        ma_type_slow_enum = MA_Type.cast(ma_type_slow)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type_slow: {str(e)}')
    
    high = quotes.high
    low = quotes.low
    
    data_len = len(high)
    if data_len < period_slow:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period_slow}')
    
    median_price = (high + low) / 2
    
    ma_fast = ma_calculate(median_price, period_fast, ma_type_fast_enum)
    ma_slow = ma_calculate(median_price, period_slow, ma_type_slow_enum)
    
    awesome = ma_fast - ma_slow
    
    if normalized:
        np.seterr(divide='ignore', invalid='ignore')
        awesome = awesome / median_price
    
    return IndicatorResult({
        'awesome': awesome
    })

