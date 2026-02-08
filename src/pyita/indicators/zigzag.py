"""zigzag(quotes, delta=0.02, depth=1, type='high_low', end_points=False)
Zig-zag indicator (pivots)."""
import numpy as np
import numba as nb

from ..indicator_result import IndicatorResult
from ..exceptions import PyTAExceptionBadParameterValue
from ..constants import PRICE_TYPE


@nb.njit(cache=True)
def find_up_corner(i_point, high, low, delta, depth):
    """Find next up corner (high pivot) in the data.
    
    Args:
        i_point: Starting index
        high: Array of high prices
        low: Array of low prices
        delta: Fraction of price change for pivot formation
        depth: Minimum distance between pivots
        
    Returns:
        Tuple of (up_corner_index, next_point_index)
    """
    n_bars = len(high)
    i_up_corner = i_point
    up_corner = high[i_point]
    i = i_point + 1
    while i < n_bars:
        if high[i] > up_corner:
            while True:
                i_up_corner = i + high[i: min(i + depth, n_bars)].argmax()
                up_corner = high[i_up_corner]
                if i_up_corner == i:
                    break
                i = i_up_corner
        elif (up_corner - low[i]) / up_corner >= delta:
            return i_up_corner, i
        i += 1

    return i_up_corner, len(high)


@nb.njit(cache=True)
def find_down_corner(i_point, high, low, delta, depth):
    """Find next down corner (low pivot) in the data.
    
    Args:
        i_point: Starting index
        high: Array of high prices
        low: Array of low prices
        delta: Fraction of price change for pivot formation
        depth: Minimum distance between pivots
        
    Returns:
        Tuple of (down_corner_index, next_point_index)
    """
    n_bars = len(high)
    i_down_corner = i_point
    down_corner = low[i_point]
    i = i_point + 1
    while i < n_bars:
        if low[i] < down_corner:
            while True:
                i_down_corner = i + low[i: min(i + depth, n_bars)].argmin()
                down_corner = low[i_down_corner]
                if i_down_corner == i:
                    break
                i = i_down_corner
        elif (high[i] - down_corner) / down_corner >= delta:
            return i_down_corner, i
        i += 1

    return i_down_corner, len(high)


@nb.njit(cache=True)
def calc_pivots(direction, high, low, delta, pivots, pivot_types, depth, checking):
    """Calculate zigzag pivots.
    
    Args:
        direction: Initial direction (1 for up, -1 for down)
        high: Array of high prices
        low: Array of low prices
        delta: Fraction of price change for pivot formation
        pivots: Output array for pivot prices (modified in place)
        pivot_types: Output array for pivot types (modified in place)
        depth: Minimum distance between pivots
        checking: If True, return index of first valid pivot
        
    Returns:
        Index of first valid pivot if checking=True, None otherwise
    """
    n_bars = len(high)

    i_point = 0
    while i_point < n_bars:

        if direction > 0:

            i_up_corner, i_point = find_up_corner(i_point, high, low, delta, depth)
            if i_point >= n_bars:
                break

            if checking and pivot_types[i_up_corner] == 1:
                return i_up_corner

            pivot_types[i_up_corner] = 1
            up_corner = high[i_up_corner]
            pivots[i_up_corner] = up_corner
            direction = -1

        else:

            i_down_corner, i_point = find_down_corner(i_point, high, low, delta, depth)
            if i_point >= n_bars:
                break

            if checking and pivot_types[i_down_corner] == -1:
                return i_down_corner

            pivot_types[i_down_corner] = -1
            down_corner = low[i_down_corner]
            pivots[i_down_corner] = down_corner
            direction = 1

    return None


@nb.njit(cache=True)
def add_last_point(pivot_types, pivots, high, low, close, delta, depth):
    """Add incomplete pivots at the end of data.
    
    Args:
        pivot_types: Array of pivot types (modified in place)
        pivots: Array of pivot prices (modified in place)
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        delta: Fraction of price change for pivot formation
        depth: Minimum distance between pivots
    """
    n_bars = len(high)

    i_last_point = 0
    for i in range(1, n_bars + 1):
        if pivot_types[n_bars - i] != 0:
            i_last_point = n_bars - i
            break

    last_pivot_type = pivot_types[i_last_point]
    if last_pivot_type == 0:
        return

    if i_last_point + depth >= n_bars:
        return

    if last_pivot_type > 0:

        v_max = high[i_last_point]
        i_min = i_last_point + depth + np.argmin(low[i_last_point + depth:])
        v_min = low[i_min]

        pivot_types[i_min] = -1
        pivots[i_min] = v_min

        if (v_max - v_min) / v_max < delta:
            return

        if i_min + depth >= n_bars:
            return

        i_current_max = i_min + np.argmax(high[i_min + depth:])
        pivot_types[i_current_max] = 1
        pivots[i_current_max] = high[i_current_max]

    else:

        v_min = low[i_last_point]
        i_max = i_last_point + depth + np.argmax(high[i_last_point + depth:])
        v_max = high[i_max]

        pivot_types[i_max] = 1
        pivots[i_max] = v_max

        if (v_max - v_min) / v_min < delta:
            return

        if i_max + depth >= n_bars:
            return

        i_current_min = i_max + np.argmin(low[i_max + depth:])
        pivot_types[i_current_min] = -1
        pivots[i_current_min] = low[i_current_min]


def get_indicator_out(quotes, delta=0.02, depth=1, type='high_low', end_points=False):
    """Calculate Zig-Zag indicator (pivots).
    
    Zig-Zag simplifies price movements by filtering out changes smaller than
    a specified threshold. It identifies pivot points where price reverses
    by at least delta fraction.
    
    Args:
        quotes: Quotes object containing OHLCV data
        delta: Fraction of price change for pivot formation (default: 0.02)
        depth: Minimum distance between H-H and L-L pivots (default: 1)
        type: Price values for pivots - 'high_low', 'close', 'open', 'high', 'low' (default: 'high_low')
        end_points: If True, add incomplete pivots at the end (default: False)
        
    Returns:
        IndicatorResult object with attributes:
            - pivots: Pivot prices (NaN where no pivot)
            - pivot_types: Pivot types (1=high, -1=low, 0=none)
            
    Raises:
        PyTAExceptionBadParameterValue: If delta <= 0, depth < 1, or type is invalid
        
    Example:
        >>> zigzag_result = zigzag(quotes, delta=0.02, depth=1)
        >>> print(zigzag_result.pivots)
        >>> print(zigzag_result.pivot_types)
    """
    if delta <= 0:
        raise PyTAExceptionBadParameterValue(f'delta must be greater than 0, got {delta}')
    
    if depth < 1:
        raise PyTAExceptionBadParameterValue(f'depth must be at least 1, got {depth}')
    
    valid_types = {'high_low', 'open', 'high', 'low', 'close'}
    if type not in valid_types:
        raise PyTAExceptionBadParameterValue(f'type must be one of {valid_types}, got {type}')
    
    close = quotes.close

    if type == 'high_low':
        high, low = quotes.high, quotes.low
    elif type in {'open', 'high', 'low', 'close'}:
        high = low = quotes[type]
    else:
        raise PyTAExceptionBadParameterValue(f'type = {type}')

    n_bars = len(close)
    pivots = np.ndarray(n_bars, dtype=PRICE_TYPE)
    pivot_types = np.zeros(n_bars, dtype=np.int8)
    pivots[:] = np.nan

    calc_pivots(-1, high, low, delta, pivots, pivot_types, depth, False)
    i_valid = calc_pivots(1, high, low, delta, pivots, pivot_types, depth, True)

    if not end_points and i_valid is not None:
        if pivot_types[i_valid] > 0:
            prev_min = low[: i_valid].min()
            if (pivots[i_valid] - prev_min) / prev_min < delta:
                i_valid += 1
        else:
            prev_max = high[: i_valid].max()
            if (prev_max - pivots[i_valid]) / prev_max < delta:
                i_valid += 1

        pivots[: i_valid] = np.nan
        pivot_types[: i_valid] = 0

    if end_points:
        add_last_point(pivot_types, pivots, high, low, close, delta, depth)

    return IndicatorResult({
        'pivots': pivots,
        'pivot_types': pivot_types
    })

