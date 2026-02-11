"""IndicatorResult class for indicator calculation results."""
import numpy as np

from .data_series import DataSeries


class IndicatorResult(DataSeries):
    """Container for indicator calculation results.
    
    This class is used to store results from technical indicators.
    It bypasses type checking and validation, accepting any dictionary
    of numpy arrays.
    
    Example:
        >>> result = IndicatorResult({'ema': np.array([1, 2, 3])})
        >>> print(result.ema)
        [1 2 3]
    """
    
    def column_types(self):
        """Return None to disable type checking.
        
        Returns:
            None: Indicates that type checking should be skipped
        """
        return None
    
    def __init__(self, data_dict):
        """Initialize IndicatorResult with a dictionary of numpy arrays.
        
        Args:
            data_dict: Dictionary containing numpy arrays
            
        Raises:
            TypeError: If data_dict is not a dictionary
            TypeError: If any value is not a numpy array
        """
        # Check that data_dict is a dictionary
        if not isinstance(data_dict, dict):
            raise TypeError(f"data_dict must be a dictionary, got {type(data_dict).__name__}")
        
        # Check that all values are numpy arrays
        for key, value in data_dict.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"All values in data_dict must be numpy arrays, but '{key}' is {type(value).__name__}")
        
        # Set data directly (bypass DataSeries.__init__ logic)
        self._data = data_dict.copy()
        self._column_types = None
        
        # Skip validation (REQUIRED_COLUMNS and ALLOWED_COLUMNS are None by default)

