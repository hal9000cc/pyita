"""vwma(quotes, period, value='close')

Volume Weighted Moving Average.

Output series: vwma (price)"""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData, PyTAExceptionDataSeriesNonFound


@nb.njit(cache=True)
def vwma_calculate(values, volume, period):
    """Calculate Volume Weighted Moving Average.
    
    Args:
        values: Array of price values
        volume: Array of volume values
        period: Period for VWMA calculation
        
    Returns:
        Array of VWMA values (first period-1 elements are NaN)
    """
    vwma = np.empty(len(values), dtype=np.float64)
    vwma[: period - 1] = np.nan

    volume_sum = volume[: period].sum()
    vwsum = (values[: period] * volume[: period]).sum()
    vwma[period - 1] = vwsum / volume_sum
    for i in range(period, len(values)):
        vwsum -= values[i - period] * volume[i - period]
        vwsum += values[i] * volume[i]
        volume_sum -= volume[i - period]
        volume_sum += volume[i]
        vwma[i] = vwsum / volume_sum

    return vwma


def get_indicator_out(quotes, period, value='close'):
    """Calculate Volume Weighted Moving Average (VWMA).
    
    VWMA is a moving average that weights each price by its volume.
    It gives more weight to periods with higher volume.
    Formula: VWMA = sum(price * volume) / sum(volume) over a rolling window
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attribute:
            - vwma: Volume Weighted Moving Average values (first period-1 elements are NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or value is not a valid price field
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes or value series is not found
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> vwma_result = vwma(quotes, period=14, value='close')
        >>> print(vwma_result.vwma)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    source_values = quotes[value]
    
    # Check if volume is present
    volume = quotes['volume']
    
    # Check minimum data requirement
    data_len = len(source_values)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    np.seterr(divide='ignore', invalid='ignore')
    vwma = vwma_calculate(source_values, volume, period)
    
    return IndicatorResult({
        'vwma': vwma
    })

