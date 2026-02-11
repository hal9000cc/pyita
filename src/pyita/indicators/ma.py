"""ma(quotes, period, value='close', ma_type='sma')

Moving average of different types: 'sma', 'ema', 'mma', 'ema0', 'mma0', 'emaw', 'mmaw'.

Output series: move_average (as source)"""
from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


def get_indicator_out(quotes, period, value='close', ma_type='sma'):
    """Calculate Moving Average of different types.
    
    This is a universal moving average indicator that supports multiple types:
    - 'sma': Simple Moving Average
    - 'ema': Exponential Moving Average (initialized via SMA)
    - 'mma': Modified EMA (initialized via SMA, alpha = 1.0 / period)
    - 'ema0': Exponential Moving Average (initialized via first data element)
    - 'mma0': Modified EMA (initialized via first data element, alpha = 1.0 / period)
    - 'emaw': EMA with dynamic-alpha warm-up (TA-Lib compatible, alpha = 2.0 / (period + 1))
    - 'mmaw': MMA (SMMA) with dynamic-alpha warm-up (TA-Lib compatible, alpha = 1.0 / period)
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', 'close', or 'volume' (default: 'close')
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0', 'emaw', 'mmaw' (default: 'sma')
        
    Returns:
        IndicatorResult object with attribute:
            - move_average: Moving average values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, value is invalid, or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> ma_result = ma(quotes, period=20, value='close', ma_type='sma')
        >>> print(ma_result.move_average)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Validate value
    valid_values = ['open', 'high', 'low', 'close', 'volume']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    # Convert ma_type string to MA_Type enum (will raise ValueError if invalid)
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    # Get source values from quotes
    source_values = quotes[value]
    
    # Check minimum data requirement
    data_len = len(source_values)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    # Calculate moving average
    out = ma_calculate(source_values, period, ma_type_enum)
    
    return IndicatorResult({
        'move_average': out
    })

