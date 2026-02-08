"""sma(quotes, period, value='close')
Simple moving average."""
from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue


def get_indicator_out(quotes, period, value='close'):
    """Calculate Simple Moving Average (SMA).
    
    SMA is the arithmetic mean of a given set of prices over a specific period.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attribute:
            - sma: Simple moving average values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or value is not a valid price field
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> sma_result = sma(quotes, period=20, value='close')
        >>> print(sma_result.sma)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    source_values = quotes[value]
    
    sma_values = ma_calculate(source_values, period, MA_Type.sma)
    
    return IndicatorResult({
        'sma': sma_values
    })

