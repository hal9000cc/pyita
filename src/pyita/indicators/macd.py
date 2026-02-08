"""macd(quotes, period_short, period_long, period_signal, ma_type='ema', ma_type_signal='sma', value='close')
Moving Average Convergence/Divergence."""
from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


def get_indicator_out(quotes, period_short, period_long, period_signal,
                      ma_type='ema', ma_type_signal='sma', value='close'):
    """Calculate Moving Average Convergence/Divergence (MACD).
    
    MACD is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices. It consists of:
    - MACD line: difference between short and long moving averages
    - Signal line: moving average of the MACD line
    - Histogram: difference between MACD and Signal lines
    
    Args:
        quotes: Quotes object containing OHLCV data
        period_short: Period for short moving average
        period_long: Period for long moving average
        period_signal: Period for signal line moving average
        ma_type: Type of moving average for MACD lines - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'ema')
        ma_type_signal: Type of moving average for signal line - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'sma')
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attributes:
            - macd: MACD line values
            - signal: Signal line values
            - hist: Histogram values (MACD - Signal)
            
    Raises:
        PyTAExceptionBadParameterValue: If periods <= 0, period_long <= period_short, value is invalid, or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> macd_result = macd(quotes, period_short=12, period_long=26, period_signal=9)
        >>> print(macd_result.macd)
        >>> print(macd_result.signal)
        >>> print(macd_result.hist)
    """
    # Validate periods
    if period_short <= 0:
        raise PyTAExceptionBadParameterValue(f'period_short must be greater than 0, got {period_short}')
    if period_long <= 0:
        raise PyTAExceptionBadParameterValue(f'period_long must be greater than 0, got {period_long}')
    if period_signal <= 0:
        raise PyTAExceptionBadParameterValue(f'period_signal must be greater than 0, got {period_signal}')
    
    # Validate period_long > period_short
    if period_long <= period_short:
        raise PyTAExceptionBadParameterValue(f'period_long ({period_long}) must be greater than period_short ({period_short})')
    
    # Validate value
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    # Convert ma_type strings to MA_Type enums (will raise ValueError if invalid)
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type: {str(e)}')
    
    try:
        ma_type_signal_enum = MA_Type.cast(ma_type_signal)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type_signal: {str(e)}')
    
    # Get source values from quotes
    source_values = quotes[value]
    
    # Check minimum data requirement
    data_len = len(source_values)
    if data_len < period_long:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period_long}')
    
    # Calculate short and long moving averages
    ema_short = ma_calculate(source_values, period_short, ma_type_enum)
    ema_long = ma_calculate(source_values, period_long, ma_type_enum)
    
    # Calculate MACD line (difference between short and long MAs)
    macd = ema_short - ema_long
    
    # Calculate signal line (moving average of MACD)
    signal = ma_calculate(macd, period_signal, ma_type_signal_enum)
    
    # Calculate histogram (MACD - Signal)
    macd_hist = macd - signal
    
    return IndicatorResult({
        'macd': macd,
        'signal': signal,
        'hist': macd_hist
    })

