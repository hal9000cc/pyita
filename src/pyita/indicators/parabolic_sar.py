"""parabolic_sar(quotes, start=0.02, maximum=0.2, increment=0.02)
Parabolic SAR."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue, PyTAExceptionTooLittleData
from ..constants import PRICE_TYPE


@nb.njit(cache=True)
def calc_paraboic(highs, lows, start, maximum, increment):
    """Calculate Parabolic SAR values.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        start: Starting acceleration factor
        maximum: Maximum acceleration factor
        increment: Increment for acceleration factor
        
    Returns:
        Tuple of (sars, signals) arrays
    """
    sars = np.empty(len(highs), dtype=np.float64)
    signals = np.zeros(len(highs), dtype=np.int8)

    is_bullish = True
    acceleration_factor = start
    sar = lows[0]
    extreme = highs[0]

    for i in range(1, len(highs)):

        sar += acceleration_factor * (extreme - sar)

        if is_bullish:

            if i > 1:
                sar = min(sar, lows[i - 1], lows[i - 2])

            if lows[i] < sar:
                is_bullish = False
                signals[i] = -1
                sar = extreme
                acceleration_factor = start
                extreme = lows[i]
            else:
                if highs[i] > extreme:
                    extreme = highs[i]
                    acceleration_factor = min(acceleration_factor + increment, maximum)

        else:

            if i > 1:
                sar = max(sar, highs[i - 1], highs[i - 2])

            if highs[i] > sar:
                is_bullish = True
                signals[i] = 1
                sar = extreme
                acceleration_factor = start
                extreme = highs[i]
            else:
                if lows[i] < extreme:
                    extreme = lows[i]
                    acceleration_factor = min(acceleration_factor + increment, maximum)

        sars[i] = sar

    # Set first elements to NaN until first signal
    for i, signal in enumerate(signals):
        sars[i] = np.nan
        signals[i] = 0
        if signal != 0:
            break

    return sars, signals


def get_indicator_out(quotes, start=0.02, maximum=0.2, increment=0.02):
    """Calculate Parabolic SAR (Stop and Reverse).
    
    Parabolic SAR is a trend-following indicator that provides entry and exit points.
    It uses an acceleration factor that increases as the trend continues, creating
    a parabolic curve. The indicator switches from bullish to bearish (or vice versa)
    when price crosses the SAR level.
    
    Args:
        quotes: Quotes object containing OHLCV data
        start: Starting acceleration factor (default: 0.02)
        maximum: Maximum acceleration factor (default: 0.2)
        increment: Increment for acceleration factor (default: 0.02)
        
    Returns:
        IndicatorResult object with attributes:
            - sar: Parabolic SAR values
            - signal: Signal values (1 for bullish, -1 for bearish, 0 otherwise)
            
    Raises:
        PyTAExceptionBadParameterValue: If start <= 0, maximum <= 0, increment <= 0, or maximum < start
        PyTAExceptionTooLittleData: If data length is less than 3
        
    Example:
        >>> sar_result = parabolic_sar(quotes, start=0.02, maximum=0.2, increment=0.02)
        >>> print(sar_result.sar)
        >>> print(sar_result.signal)
    """
    # Validate parameters
    if start <= 0:
        raise PyTAExceptionBadParameterValue(f'start must be greater than 0, got {start}')
    if maximum <= 0:
        raise PyTAExceptionBadParameterValue(f'maximum must be greater than 0, got {maximum}')
    if increment <= 0:
        raise PyTAExceptionBadParameterValue(f'increment must be greater than 0, got {increment}')
    if maximum < start:
        raise PyTAExceptionBadParameterValue(f'maximum ({maximum}) must be >= start ({start})')
    
    # Get OHLC data from quotes
    high = quotes.high
    low = quotes.low
    
    # Check minimum data requirement
    data_len = len(high)
    min_data_len = 3
    if data_len < min_data_len:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {min_data_len}')
    
    # Calculate Parabolic SAR
    parabolic_sar, signals = calc_paraboic(high, low, start, maximum, increment)
    
    return IndicatorResult({
        'sar': parabolic_sar,
        'signal': signals
    })

