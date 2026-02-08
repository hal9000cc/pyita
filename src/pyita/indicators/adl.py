"""adl(quotes, ma_period=None, ma_type='sma')
Accumulation/distribution line."""
import numpy as np

from ..indicator_result import IndicatorResult
from ..move_average import ma_calculate, MA_Type
from ..exceptions import PyTAExceptionBadParameterValue


def get_indicator_out(quotes, ma_period=None, ma_type='sma'):
    """Calculate Accumulation/Distribution Line (ADL).
    
    ADL is a volume-based indicator that uses price and volume to determine
    whether a stock is being accumulated or distributed. It calculates the
    Close Location Value (CLV) and multiplies it by volume, then accumulates
    the result.
    
    Args:
        quotes: Quotes object containing OHLCV data (volume is required)
        ma_period: Moving average period for adl_smooth value (int, optional).
                  If None, adl_smooth is not calculated (default: None)
        ma_type: Type of moving average for adl_smooth - 'sma', 'ema', 'mma',
                 'ema0', 'mma0' (default: 'sma')
        
    Returns:
        IndicatorResult object with attributes:
            - adl: Accumulation/Distribution Line values
            - adl_smooth: Smoothed ADL values (only if ma_period is not None)
            
    Raises:
        PyTAExceptionBadParameterValue: If ma_period <= 0 or ma_type is invalid
        PyTAExceptionDataSeriesNonFound: If volume is not present in quotes
        
    Example:
        >>> adl_result = adl(quotes, ma_period=14, ma_type='sma')
        >>> print(adl_result.adl)
        >>> print(adl_result.adl_smooth)
    """
    # Validate ma_period if provided
    if ma_period is not None:
        if ma_period <= 0:
            raise PyTAExceptionBadParameterValue(f'ma_period must be greater than 0, got {ma_period}')
    
    # Convert ma_type string to MA_Type enum (will raise ValueError if invalid)
    if ma_period is not None:
        try:
            ma_type_enum = MA_Type.cast(ma_type)
        except ValueError as e:
            raise PyTAExceptionBadParameterValue(str(e))
    
    high = quotes.high
    low = quotes.low
    close = quotes.close
    volume = quotes['volume']
    
    np.seterr(invalid='ignore')
    
    hl_range = high - low
    
    clv = ((close - low) - (high - close)) / hl_range
    clv[hl_range == 0] = 0
    adl = np.cumsum(clv * volume)
    
    result_data = {
        'adl': adl
    }
    
    if ma_period is not None:
        result_data['adl_smooth'] = ma_calculate(adl, ma_period, ma_type_enum)
    
    return IndicatorResult(result_data)

