"""obv(quotes)
On Balance Volume."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionDataSeriesNonFound, PyTAExceptionTooLittleData


def get_indicator_out(quotes):
    """Calculate On Balance Volume (OBV).
    
    OBV is a volume-based indicator that measures buying and selling pressure.
    It accumulates volume on up days and subtracts volume on down days.
    The OBV line should move in the same direction as the price trend.
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        
    Returns:
        IndicatorResult object with attribute:
            - obv: On Balance Volume values (cumulative sum)
            
    Raises:
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes
        PyTAExceptionTooLittleData: If data length is less than 1
        
    Example:
        >>> obv_result = obv(quotes)
        >>> print(obv_result.obv)
    """
    # Get OHLCV data from quotes
    close = quotes.close
    
    # Check if volume is present
    volume = quotes['volume']
    
    # Check minimum data requirement (at least 1 bar)
    data_len = len(close)
    if data_len < 1:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < 1')
    
    # Calculate sign of price change
    # signs[0] = 0, signs[i] = sign(close[i] - close[i-1]) for i > 0
    signs = np.hstack((0, np.sign(close[1:] - close[:-1])))
    
    # Multiply volume by sign (positive for up days, negative for down days, zero for unchanged)
    sign_volume = volume * signs
    
    # Calculate OBV as cumulative sum
    obv = np.cumsum(sign_volume)
    
    return IndicatorResult({
        'obv': obv
    })

