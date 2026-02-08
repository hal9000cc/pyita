"""mfi(quotes, period=14)
Money flow index."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData, PyTAExceptionDataSeriesNonFound


def get_indicator_out(quotes, period=14):
    """Calculate Money Flow Index (MFI).
    
    MFI is a momentum oscillator that uses both price and volume to identify
    overbought or oversold conditions. It is similar to RSI but incorporates volume.
    MFI ranges from 0 to 100, with values above 80 typically indicating overbought
    conditions and values below 20 indicating oversold conditions.
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        period: Period for MFI calculation (default: 14)
        
    Returns:
        IndicatorResult object with attribute:
            - mfi: Money Flow Index values (0-100, first period elements are NaN)
            
    Raises:
        PyTAExceptionBadParameterValue: If period <= 0
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes
        PyTAExceptionTooLittleData: If data length is less than period
        
    Example:
        >>> mfi_result = mfi(quotes, period=14)
        >>> print(mfi_result.mfi)
    """
    # Validate period
    if period <= 0:
        raise PyTAExceptionBadParameterValue(f'period must be greater than 0, got {period}')
    
    # Get OHLCV data from quotes
    high = quotes.high
    low = quotes.low
    close = quotes.close
    
    # Check if volume is present
    volume = quotes['volume']
    
    # Check minimum data requirement
    n_bars = len(high)
    if n_bars < period:
        raise PyTAExceptionTooLittleData(f'data length {n_bars} < {period}')
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    mf = typical_price * volume
    
    # Calculate sign of typical price change
    # np.diff with prepend adds the first element to maintain length
    mfz = np.sign(np.diff(typical_price, prepend=typical_price[0]))
    
    # Separate positive and negative money flow
    bx_p = mfz > 0
    mf_p = np.zeros(n_bars, dtype=np.float64)
    mf_p[bx_p] = mf[bx_p]
    
    bx_m = mfz < 0
    mf_m = np.zeros(n_bars, dtype=np.float64)
    mf_m[bx_m] = mf[bx_m]
    
    # Calculate sum of positive and negative money flow over period using convolution
    # np.convolve with default mode='full' returns len(a) + len(b) - 1, then we take [:n_bars]
    weights = np.ones(period, dtype=np.float64)
    mf_sum_p = np.convolve(mf_p, weights)[:n_bars]
    mf_sum_m = np.convolve(mf_m, weights)[:n_bars]
    
    # Calculate MFI
    np.seterr(divide='ignore', invalid='ignore')
    mfi = 100.0 * mf_sum_p / (mf_sum_p + mf_sum_m)
    # Handle division by zero
    mfi[mf_sum_p + mf_sum_m == 0] = 0
    # Set first period elements to NaN
    mfi[:period] = np.nan
    
    return IndicatorResult({
        'mfi': mfi
    })

