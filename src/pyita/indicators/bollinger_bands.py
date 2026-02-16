"""bollinger_bands(quotes, period=20, deviation=2, deviation_up=None, deviation_down=None, ma_type='sma', value='close')

Bollinger bands.

Output series: mid_line (price), up_line (price), down_line (price), z_score"""

import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue
from ..helpers import validate_value_par


@nb.njit(cache=True)
def calc_std_deviations(values, period):
    """Calculate rolling standard deviations.
    
    Args:
        values: Array of price values
        period: Period for standard deviation calculation
        
    Returns:
        Array of standard deviations
    """
    values_len = len(values)
    result = np.empty(values_len)
    result[:period - 1] = np.nan
    
    for i in range(period, values_len + 1):
        result[i - 1] = values[i - period: i].std()
    
    return result


def get_indicator_out(quotes, period=20, deviation=2, deviation_up=None, deviation_down=None, ma_type='sma', value='close'):
    """Calculate Bollinger Bands indicator.
    
    Bollinger Bands consist of a middle line (moving average) and two bands
    above and below it, positioned at a specified number of standard deviations.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation (default: 20)
        deviation: Number of standard deviations for bands (default: 2, used when deviation_up/deviation_down are None)
        deviation_up: Number of standard deviations for upper band (default: None, uses deviation if not set)
        deviation_down: Number of standard deviations for lower band (default: None, uses deviation if not set)
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attributes:
            - mid_line: Middle line (moving average)
            - up_line: Upper band
            - down_line: Lower band
            - z_score: Z-score (distance from middle in standard deviations)
            
    Raises:
        PyTAExceptionBadParameterValue: If parameters are invalid
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> bb = bollinger_bands(quotes, period=20, deviation=2)
        >>> print(bb.mid_line)
        >>> print(bb.up_line)
        >>> print(bb.down_line)
        >>> bb = bollinger_bands(quotes, period=20, deviation_up=2.5, deviation_down=1.5)
        >>> print(bb.up_line)
        >>> print(bb.down_line)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    if deviation_up is None and deviation_down is None:
        if deviation <= 0:
            raise PyTAExceptionBadParameterValue(f'deviation must be greater than 0, got {deviation}')
    
    if deviation_up is not None:
        if deviation_up <= 0:
            raise PyTAExceptionBadParameterValue(f'deviation_up must be greater than 0, got {deviation_up}')
    
    if deviation_down is not None:
        if deviation_down <= 0:
            raise PyTAExceptionBadParameterValue(f'deviation_down must be greater than 0, got {deviation_down}')
    
    validate_value_par(value, allow_volume=True)
    
    source_values = quotes[value]
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    mid_line = ma_calculate(source_values, period, ma_type_enum)
    std_deviations = calc_std_deviations(source_values, period)
    
    up_deviation = deviation_up if deviation_up is not None else deviation
    down_deviation = deviation_down if deviation_down is not None else deviation
    
    up_line = mid_line + (std_deviations * up_deviation)
    down_line = mid_line - (std_deviations * down_deviation)
    
    np.seterr(divide='ignore', invalid='ignore')
    z_score = (source_values - mid_line) / std_deviations
    z_score[std_deviations == 0] = 0
    
    return IndicatorResult({
        'mid_line': mid_line,
        'up_line': up_line,
        'down_line': down_line,
        'z_score': z_score
    })

