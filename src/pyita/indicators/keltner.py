"""keltner(quotes, period=10, multiplier=1, period_atr=10, ma_type='ema', ma_type_atr='mma')
Keltner channel."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from . import atr


def get_indicator_out(quotes, period=10, multiplier=1, period_atr=10, ma_type='ema', ma_type_atr='mma'):
    """Calculate Keltner Channel.
    
    Keltner Channel is a volatility-based indicator that uses ATR to set channel
    boundaries around a moving average. It consists of three lines:
    - Middle line: Moving average of close prices
    - Upper line: Middle line + ATR * multiplier
    - Lower line: Middle line - ATR * multiplier
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for middle line moving average (default: 10)
        multiplier: Multiplier for ATR (default: 1)
        period_atr: Period for ATR calculation (default: 10)
        ma_type: Type of moving average for middle line - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'ema')
        ma_type_atr: Type of moving average for ATR - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'mma')
        
    Returns:
        IndicatorResult object with attributes:
            - mid_line: Middle line (moving average) values
            - up_line: Upper channel line values
            - down_line: Lower channel line values
            - width: Channel width as percentage of middle line
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, multiplier <= 0, period_atr <= 0, or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> keltner_result = keltner(quotes, period=10, multiplier=1, period_atr=10)
        >>> print(keltner_result.mid_line)
        >>> print(keltner_result.up_line)
        >>> print(keltner_result.down_line)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Validate multiplier
    if multiplier <= 0:
        raise PyTAExceptionBadParameterValue(f'multiplier must be greater than 0, got {multiplier}')
    
    # Validate period_atr
    if period_atr <= 0:
        raise PyTAExceptionBadParameterValue(f'period_atr must be greater than 0, got {period_atr}')
    
    # Convert ma_type strings to MA_Type enums (will raise ValueError if invalid)
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type: {str(e)}')
    
    try:
        ma_type_atr_enum = MA_Type.cast(ma_type_atr)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(f'ma_type_atr: {str(e)}')
    
    # Get close data from quotes
    close = quotes.close
    
    # Check minimum data requirement
    max_period = max(period, period_atr)
    data_len = len(close)
    if data_len < max_period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {max_period}')
    
    # Calculate ATR
    atr_result = atr.get_indicator_out(quotes, smooth=period_atr, ma_type=ma_type_atr)
    atr_values = atr_result.atr
    
    # Calculate middle line (moving average of close)
    mid_line = ma_calculate(close, period, ma_type_enum)
    
    # Calculate channel lines
    up_line = mid_line + atr_values * multiplier
    down_line = mid_line - atr_values * multiplier
    
    # Calculate width (channel width as percentage of middle line)
    np.seterr(divide='ignore', invalid='ignore')
    width = (up_line - down_line) / mid_line
    # Handle division by zero
    width[mid_line == 0] = 0
    
    return IndicatorResult({
        'mid_line': mid_line,
        'up_line': up_line,
        'down_line': down_line,
        'width': width
    })

