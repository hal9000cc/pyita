"""ema(quotes, period, value='close')
Exponential moving average."""
from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue


def get_indicator_out(quotes, period, value='close'):
    """Calculate Exponential Moving Average (EMA).
    
    EMA is a type of moving average that places greater weight on recent data points.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        DataSeries object with attribute:
            - ema: Exponential moving average values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or value is not a valid price field
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> ema_result = ema(quotes, period=12, value='close')
        >>> print(ema_result.ema)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Validate value
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    # Get source values from quotes
    source_values = quotes[value]
    
    # Calculate EMA
    ema_values = ma_calculate(source_values, period, MA_Type.ema)
    
    return IndicatorResult({
        'ema': ema_values
    })

