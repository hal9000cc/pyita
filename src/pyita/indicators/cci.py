"""cci(quotes, period=20)
Commodity channel index."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData


@nb.njit(cache=True)
def calc_mad(typical_price, sma_typical_price, period):
    """Calculate Mean Absolute Deviation (MAD).
    
    Args:
        typical_price: Array of typical prices
        sma_typical_price: Array of SMA values of typical prices
        period: Period for MAD calculation
        
    Returns:
        Array of MAD values
    """
    values_len = len(sma_typical_price)

    mad = np.empty(values_len, dtype=np.float64)
    mad[:period - 1] = 0
    for i in range(period, values_len + 1):
        mad[i - 1] = np.abs(typical_price[i - period: i] - sma_typical_price[i - 1]).sum() / period

    return mad


def get_indicator_out(quotes, period=20):
    """Calculate Commodity Channel Index (CCI).
    
    CCI is a momentum-based oscillator used to identify cyclical trends in commodities.
    It measures the deviation of price from its statistical mean, normalized by the
    Mean Absolute Deviation (MAD).
    
    Args:
        quotes: Quotes object containing OHLCV data
        period: Period for CCI calculation (default: 20)
        
    Returns:
        IndicatorResult object with attribute:
            - cci: Commodity Channel Index values
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> cci_result = cci(quotes, period=20)
        >>> print(cci_result.cci)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Get OHLC data from quotes
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    # Check minimum data requirement
    data_len = len(high)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate SMA of typical price
    sma_typical_price = ma_calculate(typical_price, period, MA_Type.sma)
    
    # Calculate Mean Absolute Deviation (MAD)
    mad = calc_mad(typical_price, sma_typical_price, period)
    
    # Calculate CCI
    # CCI = (typical_price - sma_typical_price) / (mad * 0.015)
    np.seterr(invalid='ignore')
    cci = (typical_price - sma_typical_price) / mad / 0.015
    # Handle division by zero
    cci[mad == 0] = 0
    
    return IndicatorResult({
        'cci': cci
    })

