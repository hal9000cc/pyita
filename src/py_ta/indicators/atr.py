"""atr(quotes, smooth=14, ma_type='mma')
Average True Range."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue
from ..constants import PRICE_TYPE


def get_indicator_out(quotes, smooth=14, ma_type='mma'):
    """Calculate Average True Range (ATR).
    
    ATR is a volatility indicator that measures the degree of price volatility.
    It calculates the True Range (TR) and then applies a moving average to it.
    
    Args:
        quotes: Quotes object containing OHLCV data
        smooth: Period for moving average calculation (default: 14)
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'mma')
        
    Returns:
        IndicatorResult object with attributes:
            - tr: True Range values
            - atr: Average True Range values
            - atrp: ATR as percentage of close price
            
    Raises:
        PyTAExceptionBadParameterValue: If smooth <= 0 or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is insufficient
        
    Example:
        >>> atr_result = atr(quotes, smooth=14, ma_type='mma')
        >>> print(atr_result.atr)
        >>> print(atr_result.tr)
        >>> print(atr_result.atrp)
    """
    # Validate smooth
    if smooth <= 0:
        raise PyTAExceptionBadParameterValue(f'smooth must be greater than 0, got {smooth}')
    
    # Convert ma_type string to MA_Type enum (will raise ValueError if invalid)
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    # Get OHLC data from quotes
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    # Calculate True Range components
    # range_current = high - low
    range_current = high - low
    
    # range_prev_high = [0] + abs(close[:-1] - high[1:])
    range_prev_high = np.hstack((
        np.zeros(1, dtype=PRICE_TYPE),
        np.abs(close[:-1] - high[1:])
    ))
    
    # range_prev_low = [0] + abs(close[:-1] - low[1:])
    range_prev_low = np.hstack((
        np.zeros(1, dtype=PRICE_TYPE),
        np.abs(close[:-1] - low[1:])
    ))
    
    # True Range = max(range_current, range_prev_high, range_prev_low)
    tr = np.maximum(range_current, np.maximum(range_prev_high, range_prev_low))
    
    # Calculate ATR (moving average of TR)
    atr = ma_calculate(tr, smooth, ma_type_enum)
    
    # Calculate ATRP (ATR as percentage of close)
    atrp = atr / close * 100
    
    return IndicatorResult({
        'tr': tr,
        'atr': atr,
        'atrp': atrp
    })

