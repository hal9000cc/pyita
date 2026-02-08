"""rsi(quotes, period, ma_type='mma', value='close')
Relative Strength Index."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue
from ..constants import PRICE_TYPE


def rsi_calculate(source_values, period, ma_type):
    """Calculate RSI from source values.
    
    Args:
        source_values: Array of price values
        period: Period for RSI calculation
        ma_type: MA_Type enum for smoothing
        
    Returns:
        Array with RSI values (first element is NaN)
    """
    U = np.diff(source_values)
    D = -U

    U[U < 0] = 0
    D[D < 0] = 0

    U_smooth = ma_calculate(U, period, ma_type)
    D_smooth = ma_calculate(D, period, ma_type)

    divider = U_smooth + D_smooth
    res = U_smooth / divider * 100
    res[divider == 0] = 100

    return np.hstack((np.nan, res))


def get_indicator_out(quotes, period, ma_type='mma', value='close'):
    """Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    It ranges from 0 to 100, with values above 70 typically indicating overbought
    conditions and values below 30 indicating oversold conditions.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for RSI calculation
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'mma')
        value: Price field to use - 'open', 'high', 'low', or 'close' (default: 'close')
        
    Returns:
        IndicatorResult object with attribute:
            - rsi: RSI values (first element is NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, value is invalid, or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If the specified value series is not found
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> rsi_result = rsi(quotes, period=14, ma_type='mma', value='close')
        >>> print(rsi_result.rsi)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    valid_values = ['open', 'high', 'low', 'close']
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    source_values = quotes[value]
    
    if len(source_values) == 0:
        out = np.zeros(0, dtype=PRICE_TYPE)
    else:
        out = rsi_calculate(source_values, period, ma_type_enum)
    
    return IndicatorResult({
        'rsi': out
    })

