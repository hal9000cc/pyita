"""Quotes class for OHLCV data."""
from .core import DataSeries
from .constants import PRICE_TYPE, VOLUME_TYPE, TIME_TYPE


class Quotes(DataSeries):
    """Container for OHLCV (Open, High, Low, Close, Volume) quote data.
    
    Supports multiple initialization methods:
    - Quotes(open, high, low, close)
    - Quotes(open, high, low, close, volume)
    - Quotes(open, high, low, close, volume, time)
    - Quotes(pandas_dataframe)
    - Quotes(**{'open': ..., 'high': ..., ...})
    
    Attributes:
        open: Array of opening prices
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        volume: Array of volumes (optional)
        time: Array of timestamps (optional)
    
    Example:
        >>> import numpy as np
        >>> open_data = np.array([100, 102, 101])
        >>> high_data = np.array([105, 106, 104])
        >>> low_data = np.array([99, 101, 100])
        >>> close_data = np.array([102, 103, 101])
        >>> quotes = Quotes(open_data, high_data, low_data, close_data)
        >>> print(quotes.close)
        [102 103 101]
    """
    
    # Column validation
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
    ALLOWED_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'time']
    
    def column_types(self):
        """Return dictionary mapping column names to their data types.
        
        Returns:
            dict: Dictionary with column names and their types
        """
        return {
            'open': PRICE_TYPE,
            'high': PRICE_TYPE,
            'low': PRICE_TYPE,
            'close': PRICE_TYPE,
            'volume': VOLUME_TYPE,
            'time': TIME_TYPE,
        }
    
    def __init__(self, *args, **kwargs):
        """Initialize Quotes with OHLCV data.
        
        Args:
            *args: Can be:
                - (open, high, low, close)
                - (open, high, low, close, volume)
                - (open, high, low, close, volume, time)
                - (pandas_dataframe,)
            **kwargs: Named arguments for explicit initialization
        
        Raises:
            PyTAExceptionBadSeriesData: If arguments are invalid or incompatible
        """
        # Call parent constructor to process args and kwargs
        super().__init__(*args, **kwargs)

