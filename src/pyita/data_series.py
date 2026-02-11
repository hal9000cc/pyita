import abc
from datetime import date, datetime

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .exceptions import PyTAExceptionBadParameterValue, PyTAExceptionBadSeriesData, PyTAExceptionDataSeriesNonFound


class DataSeries(abc.ABC):
    """Base class for quotes and indicator results.
    
    Acts as a dictionary with attribute-style access.
    
    Example:
        >>> data = DataSeries({'value': [1, 2, 3], 'time': [...]})
        >>> print(data.value)
        [1, 2, 3]
    """
    
    # Column validation attributes
    # Set to None to skip validation, or list of column names to validate
    REQUIRED_COLUMNS = None  # Required columns (None = skip check)
    ALLOWED_COLUMNS = None   # Allowed columns (None = skip check)
    
    @abc.abstractmethod
    def column_types(self):
        """Return dictionary mapping column names to their data types.
        
        Returns:
            dict: Dictionary with column names as keys and data types as values
                Example: {'open': float, 'high': float, 'time': 'datetime64[ms]'}
        """
        pass
    
    def __init__(self, *args, **kwargs):
        """Initialize DataSeries with data from args and kwargs.
        
        Args:
            *args: Positional arguments:
                - If single argument with 'columns' attribute (pandas DataFrame): process as DataFrame
                - Otherwise: process as positional data (args[0] -> first column, args[1] -> second, etc.)
            **kwargs: Named arguments for explicit column specification
        """
        # Get column types from subclass
        self._column_types = self.column_types()
        
        # Initialize data dictionary
        self._data = {}
        
        # Process arguments
        if len(args) == 1 and hasattr(args[0], 'columns'):
            # Pandas DataFrame
            if not HAS_PANDAS:
                raise PyTAExceptionBadParameterValue(
                    "pandas is required to process DataFrame. Install it with: pip install pandas"
                )
            self._process_pandas_dataframe(args[0])
        elif len(args) == 1 and self._is_ccxt_format(args[0]):
            # CCXT format: list of lists [[timestamp, open, high, low, close, volume], ...]
            self._process_ccxt_format(args[0])
        else:
            # Process positional arguments
            column_names = ['open', 'high', 'low', 'close', 'volume', 'time']
            if len(args) > len(column_names):
                raise PyTAExceptionBadParameterValue(
                    f"Too many positional arguments: {len(args)}, maximum {len(column_names)} allowed"
                )
            for i, arg in enumerate(args):
                self._add_data(column_names[i], arg)
        
        # Process keyword arguments
        for key, value in kwargs.items():
            self._add_data(key, value)
        
        # Validate data
        self._validate_data()
    
    def _add_data(self, key, data):
        """Add data to internal dictionary with type conversion.
        
        Args:
            key: Column name
            data: Data to add (can be list, numpy array, int, float, datetime, etc.)
            
        Raises:
            PyTAExceptionBadParameterValue: If key already exists or type not found in column_types
        """
        # Check if key already exists
        if key in self._data:
            raise PyTAExceptionBadParameterValue(f"Column '{key}' already exists")
        
        # Get data type for this column
        # If column_types is None, skip type checking and conversion
        if self._column_types is None:
            # For IndicatorResult: just store the data as-is (should already be numpy array)
            if not isinstance(data, np.ndarray):
                raise PyTAExceptionBadParameterValue(f"Data must be numpy array when column_types is None, got {type(data).__name__}")
            self._data[key] = data
            return
        
        if key not in self._column_types:
            raise PyTAExceptionBadParameterValue(f"Unknown column type for '{key}'")
        
        dtype = self._column_types[key]
        
        # Determine conversion strategy based on dtype characteristics
        if isinstance(dtype, str) and dtype.startswith('datetime64'):
            # Datetime type - convert to target datetime64
            converted = self._convert_to_datetime(data, dtype)
        elif dtype in (float, int) or (isinstance(dtype, type) and issubclass(dtype, (float, int))):
            # Numeric type - convert to numeric array
            converted = self._convert_to_numeric(data, dtype)
        else:
            # Default: try to convert to numpy array with specified dtype
            converted = np.array(data, dtype=dtype)
        
        self._data[key] = converted
    
    def _convert_to_numeric(self, data, dtype):
        """Convert data to numeric numpy array.
        
        Args:
            data: Data to convert (list, tuple, numpy array, scalar)
            dtype: Target numeric type (float, int, etc.)
            
        Returns:
            numpy.ndarray: Numeric array with specified dtype
        """
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return data.astype(dtype)
        elif isinstance(data, (int, float)):
            return np.array([data], dtype=dtype)
        else:
            return np.array(data, dtype=dtype)
    
    def _convert_to_datetime(self, data, target_dtype):
        """Convert data to target datetime64 type.
        
        Args:
            data: Data to convert (list, tuple, numpy array, datetime, date, etc.)
            target_dtype: Target datetime64 type string (e.g., 'datetime64[ms]', 'datetime64[us]')
            
        Returns:
            numpy.ndarray: Datetime array with target dtype
        """
        # Extract unit from target_dtype (e.g., 'ms' from 'datetime64[ms]')
        unit = 'ms'  # Default
        if '[' in target_dtype and ']' in target_dtype:
            unit = target_dtype.split('[')[1].split(']')[0]
        
        if isinstance(data, (list, tuple)):
            converted = np.array([self._convert_single_datetime(item, unit) for item in data], dtype=target_dtype)
        elif isinstance(data, np.ndarray):
            if data.dtype.kind == 'M':  # Already datetime
                converted = data.astype(target_dtype)
            else:
                converted = np.array([self._convert_single_datetime(item, unit) for item in data], dtype=target_dtype)
        elif isinstance(data, (datetime, date)):
            converted = np.array([self._convert_single_datetime(data, unit)], dtype=target_dtype)
        else:
            converted = np.array([self._convert_single_datetime(data, unit)], dtype=target_dtype)
        return converted
    
    def _convert_single_datetime(self, value, unit='ms'):
        """Convert single datetime value to datetime64 with specified unit.
        
        Args:
            value: datetime, date, datetime64, or timestamp
            unit: Datetime unit (e.g., 'ms', 'us', 's')
            
        Returns:
            numpy.datetime64: datetime64 with specified unit
        """
        if isinstance(value, datetime):
            return np.datetime64(value, unit)
        elif isinstance(value, date):
            # Convert date to start of day (00:00:00)
            dt = datetime.combine(value, datetime.min.time())
            return np.datetime64(dt, unit)
        elif isinstance(value, np.datetime64):
            return value.astype(f'datetime64[{unit}]')
        elif HAS_PANDAS and isinstance(value, pd.Timestamp):
            return np.datetime64(value, unit)
        else:
            # Try to convert as timestamp
            return np.datetime64(value, unit)
    
    def _is_ccxt_format(self, data):
        """Check if data is in CCXT format (list of lists).
        
        Args:
            data: Data to check
            
        Returns:
            bool: True if data is in CCXT format (list of lists), False otherwise
        """
        if not isinstance(data, (list, tuple)):
            return False
        if len(data) == 0:
            return False
        if not isinstance(data[0], (list, tuple)):
            return False
        return True
    
    def _process_ccxt_format(self, ohlcv_list):
        """Process CCXT format: list of lists [[timestamp, open, high, low, close, volume], ...].
        
        Args:
            ohlcv_list: List of lists in CCXT format
                Each inner list: [timestamp, open, high, low, close, volume]
        """
        if not ohlcv_list:
            raise PyTAExceptionBadSeriesData("CCXT format list cannot be empty")
        
        # Extract columns from CCXT format
        # Format: [[timestamp, open, high, low, close, volume], ...]
        time_data = [row[0] for row in ohlcv_list]
        open_data = [row[1] for row in ohlcv_list]
        high_data = [row[2] for row in ohlcv_list]
        low_data = [row[3] for row in ohlcv_list]
        close_data = [row[4] for row in ohlcv_list]
        
        # Add required columns
        self._add_data('open', open_data)
        self._add_data('high', high_data)
        self._add_data('low', low_data)
        self._add_data('close', close_data)
        
        # Add optional columns if present
        if len(ohlcv_list[0]) >= 6:
            volume_data = [row[5] for row in ohlcv_list]
            self._add_data('volume', volume_data)
        
        if len(ohlcv_list[0]) >= 1:
            self._add_data('time', time_data)
    
    def _process_pandas_dataframe(self, df):
        """Process pandas DataFrame and extract columns.
        
        Args:
            df: pandas DataFrame with OHLCV columns (case-insensitive matching)
            
        Raises:
            PyTAExceptionBadParameterValue: If pandas is not installed
        """
        if not HAS_PANDAS:
            raise PyTAExceptionBadParameterValue(
                "pandas is required to process DataFrame. Install it with: pip install pandas"
            )
        # Column name mapping (case-insensitive)
        column_mapping = {}
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for expected_col in ['open', 'high', 'low', 'close', 'volume', 'time']:
            if expected_col.lower() in df_columns_lower:
                actual_col = df_columns_lower[expected_col.lower()]
                column_mapping[expected_col] = actual_col
        
        # Add data for each found column
        for expected_col, actual_col in column_mapping.items():
            self._add_data(expected_col, df[actual_col].values)
    
    def _check_lengths(self, columns):
        """Check if all specified columns have the same length.
        
        Args:
            columns: List of column names to check
            
        Returns:
            bool: True if all columns have the same length, False otherwise
        """
        if not columns:
            return True
        
        lengths = [len(self._data[col]) for col in columns if col in self._data]
        
        if not lengths:
            return True
        
        return all(length == lengths[0] for length in lengths)
    
    def _validate_data(self):
        """Validate data according to required and allowed columns, and check lengths.
        
        Raises:
            PyTAExceptionBadSeriesData: If validation fails
        """
        # Check allowed columns first (if specified)
        if self.ALLOWED_COLUMNS is not None:
            for col in self._data.keys():
                if col not in self.ALLOWED_COLUMNS:
                    raise PyTAExceptionBadSeriesData(f"Unknown column: {col}")
        
        # Check required columns (if specified)
        if self.REQUIRED_COLUMNS is not None:
            for col in self.REQUIRED_COLUMNS:
                if col not in self._data:
                    raise PyTAExceptionBadSeriesData(f"Missing required column: {col}")
        
        # Check that all columns have the same length (if more than one column)
        all_columns = list(self._data.keys())
        if len(all_columns) > 1:
            if not self._check_lengths(all_columns):
                lengths = {col: len(self._data[col]) for col in all_columns}
                length_details = ', '.join([f"{col}={length}" for col, length in lengths.items()])
                raise PyTAExceptionBadSeriesData(f"Arrays have different lengths: {length_details}")
    
    def __getitem__(self, key):
        """Get data series by key using bracket notation or slice.
        
        Args:
            key: Can be:
                - String: Series name (e.g., 'open', 'close', 'volume')
                - slice: Slice object (e.g., slice(1, 10), slice(None, 10))
                - int: Index (returns object with single element)
            
        Returns:
            If key is string: Value from internal dictionary (numpy array)
            If key is slice/int: New DataSeries object with sliced data
            
        Raises:
            PyTAExceptionDataSeriesNonFound: If series name not found
            IndexError: If int index is out of range
        """
        if isinstance(key, str):
            if key not in self._data:
                raise PyTAExceptionDataSeriesNonFound(key)
            return self._data[key]
        
        if isinstance(key, (slice, int)):
            return self._create_sliced(key)
        
        raise TypeError(f"Unsupported key type: {type(key).__name__}")
    
    def _create_sliced(self, key):
        """Create a new DataSeries object with sliced data.
        
        Args:
            key: slice object or int index
            
        Returns:
            New DataSeries object (same type as self) with sliced arrays
            
        Raises:
            IndexError: If int index is out of range
        """
        if not self._data:
            return self._create_empty()
        
        first_array = next(iter(self._data.values()))
        data_len = len(first_array)
        
        if isinstance(key, int):
            if key < 0:
                key = data_len + key
            if key < 0 or key >= data_len:
                raise IndexError(f"Index {key} is out of range for length {data_len}")
            key = slice(key, key + 1)
        
        sliced_data = {}
        for col_name, arr in self._data.items():
            sliced_data[col_name] = arr[key]
        
        return self._create_from_dict(sliced_data)
    
    def _create_empty(self):
        """Create an empty DataSeries object of the same type.
        
        Returns:
            New empty DataSeries object (same type as self)
        """
        return self._create_from_dict({})
    
    def _create_from_dict(self, data_dict):
        """Create a new DataSeries object from a dictionary of arrays.
        
        This method should be overridden in subclasses for optimal performance.
        Default implementation creates object and sets attributes directly.
        
        Args:
            data_dict: Dictionary of column names to numpy arrays
            
        Returns:
            New DataSeries object (same type as self)
        """
        # Create new instance without calling __init__
        new_obj = type(self).__new__(type(self))
        
        # Set attributes directly (bypass validation since data is already valid)
        new_obj._data = data_dict
        new_obj._column_types = self._column_types
        
        return new_obj
    
    @property
    def writeable(self):
        """Get writeable flag from first array.
        
        Returns:
            bool or None: Writeable flag of first array, or None if object is empty
            
        Example:
            >>> quotes = Quotes(open, high, low, close)
            >>> print(quotes.writeable)
            True
        """
        if not self._data:
            return None
        first_array = next(iter(self._data.values()))
        return first_array.flags.writeable
    
    @writeable.setter
    def writeable(self, value):
        """Set writeable flag for all arrays.
        
        Args:
            value: Boolean value to set for writeable flag
            
        Example:
            >>> quotes = Quotes(open, high, low, close)
            >>> quotes.writeable = False
            >>> quotes.close[0] = 999  # Will raise ValueError
        """
        for arr in self._data.values():
            arr.flags.writeable = value
    
    def __getattr__(self, name):
        """Get attribute from internal data dictionary.
        
        Args:
            name: Attribute name
            
        Returns:
            Value from internal dictionary
            
        Raises:
            AttributeError: If attribute not found
        """
        if name.startswith('_'):
            # Avoid infinite recursion for internal attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __repr__(self):
        """String representation of DataSeries."""
        keys = ', '.join(self._data.keys())
        return f"{type(self).__name__}({keys})"

