"""adx(quotes, period=14, smooth=14, ma_type='mma')
Average directional movement index."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from . import atr


def get_indicator_out(quotes, period=14, smooth=14, ma_type='mma'):
    """Calculate Average Directional Movement Index (ADX).
    
    ADX is a trend strength indicator that measures the strength of a trend
    regardless of direction. It uses the Directional Movement (DM) and True Range
    to calculate Directional Indicators (DI) and then smooths them to get ADX.
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for Directional Indicators (DI) calculation (default: 14)
        smooth: Period for ADX smoothing (default: 14)
        ma_type: Type of moving average - 'sma', 'ema', 'mma', 'ema0', 'mma0' (default: 'mma')
        
    Returns:
        IndicatorResult object with attributes:
            - adx: Average Directional Movement Index values
            - p_di: Positive Directional Indicator values
            - m_di: Negative Directional Indicator values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0, smooth <= 0, or ma_type is invalid
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> adx_result = adx(quotes, period=14, smooth=14, ma_type='mma')
        >>> print(adx_result.adx)
        >>> print(adx_result.p_di)
        >>> print(adx_result.m_di)
    """
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    if smooth <= 0:
        raise PyTAExceptionBadParameterValue(f'smooth must be greater than 0, got {smooth}')
    
    try:
        ma_type_enum = MA_Type.cast(ma_type)
    except ValueError as e:
        raise PyTAExceptionBadParameterValue(str(e))
    
    high = quotes.high
    low = quotes.low
    
    data_len = len(high)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    # Calculate Directional Movement
    p_dm = np.hstack([np.nan, np.diff(high)])
    m_dm = np.hstack([np.nan, -np.diff(low)])
    
    # Zero out DM when conditions are not met
    bx_zero_p_dm = (p_dm <= m_dm) | (p_dm < 0)
    bx_zero_m_dm = (m_dm <= p_dm) | (m_dm < 0)
    p_dm[bx_zero_p_dm] = 0
    m_dm[bx_zero_m_dm] = 0
    
    # Calculate ATR for normalization
    atr_result = atr.get_indicator_out(quotes, smooth=period, ma_type=ma_type)
    atr_values = atr_result.atr
    
    # Calculate Directional Indicators (DI)
    p_di = 100 * ma_calculate(p_dm, period, ma_type_enum) / atr_values
    m_di = 100 * ma_calculate(m_dm, period, ma_type_enum) / atr_values
    
    # Calculate Directional Index (DX)
    np.seterr(divide='ignore', invalid='ignore')
    dxi = 100 * np.abs(p_di - m_di) / (p_di + m_di)
    dxi[p_di + m_di == 0] = 0
    
    # Calculate ADX (smoothed DX)
    adx = ma_calculate(dxi, smooth, ma_type_enum)
    
    return IndicatorResult({
        'adx': adx,
        'p_di': p_di,
        'm_di': m_di
    })

