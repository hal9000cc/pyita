"""volume_osc(quotes, period_short=5, period_long=10, ma_type='ema')
Volume oscillator."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData, PyTAExceptionDataSeriesNonFound


def get_indicator_out(quotes, period_short=5, period_long=10, ma_type='ema'):
    """Calculate Volume Oscillator.
    
    Volume Oscillator measures the difference between short and long period
    moving averages of volume, expressed as a percentage of the long period average.
    Formula: osc = (vol_short - vol_long) / vol_long * 100
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        period_short: Period for short moving average (default: 5)
        period_long: Period for long moving average (default: 10)
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'ema')
        
    Returns:
        IndicatorResult object with attribute:
            - osc: Volume Oscillator values
            
    Raises:
        PyTAExceptionBadParameterValue: If period_short <= 0, period_long <= 0, period_long <= period_short, or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes
        PyTAExceptionTooLittleData: If data length is less than period_long
        
    Example:
        >>> vosc_result = volume_osc(quotes, period_short=5, period_long=10)
        >>> print(vosc_result.osc)
    """
    if period_short <= 0:
        raise PyTAExceptionBadParameterValue(f'period_short must be greater than 0, got {period_short}')
    
    if period_long <= 0:
        raise PyTAExceptionBadParameterValue(f'period_long must be greater than 0, got {period_long}')
    
    if period_long <= period_short:
        raise PyTAExceptionBadParameterValue(f'period_long ({period_long}) must be greater than period_short ({period_short})')
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    # Check if volume is present
    volume = quotes['volume']
    
    # Check minimum data requirement
    data_len = len(volume)
    if data_len < period_long:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period_long}')
    
    vol_short = ma_calculate(volume, period_short, ma_type_enum)
    vol_long = ma_calculate(volume, period_long, ma_type_enum)
    
    np.seterr(divide='ignore', invalid='ignore')
    osc = (vol_short - vol_long) / vol_long * 100
    
    return IndicatorResult({
        'osc': osc
    })

