"""tema(quotes, period, value='close')
Triple Exponential Moving Average."""
from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue


def get_indicator_out(quotes, period, value='close'):
    """Calculate Triple Exponential Moving Average (TEMA).
    
    TEMA applies exponential smoothing three times to reduce lag.
    Formula: TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA1 = EMA(source), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for moving average calculation
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attribute:
            - tema: Triple exponential moving average values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0 or value is not a valid price field
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is less than required
        
    Example:
        >>> tema_result = tema(quotes, period=14, value='close')
        >>> print(tema_result.tema)
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
    
    tema = (ema1 * 3) - (ema2 * 3) + ema3
    
    return IndicatorResult({
        'tema': tema
    })

