"""vwap(quotes)

Volume Weighted Average Price.

Output series: vwap (price)"""
import numpy as np

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionDataSeriesNonFound, PyTAExceptionTooLittleData


def get_indicator_out(quotes):
    """Calculate Volume Weighted Average Price (VWAP).
    
    VWAP is the average price of a security weighted by volume. It provides
    a benchmark for traders to assess the quality of their execution.
    Formula: VWAP = cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        
    Returns:
        IndicatorResult object with attribute:
            - vwap: Volume Weighted Average Price values
            
    Raises:
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes
        PyTAExceptionTooLittleData: If data length is less than 1
        
    Example:
        >>> vwap_result = vwap(quotes)
        >>> print(vwap_result.vwap)
    """
    # Get OHLCV data from quotes
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    # Check if volume is present
    volume = quotes['volume']
    
    # Check minimum data requirement (at least 1 bar)
    data_len = len(close)
    if data_len < 1:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < 1')
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP
    np.seterr(divide='ignore', invalid='ignore')
    typical_price_volume = typical_price * volume
    volume_sum = np.cumsum(volume)
    vwap = np.cumsum(typical_price_volume) / volume_sum
    
    return IndicatorResult({
        'vwap': vwap
    })

