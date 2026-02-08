"""Core data structures for pyita."""
import abc
from datetime import date, datetime

import numpy as np
import pandas as pd

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
            self._process_pandas_dataframe(args[0])
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
        elif isinstance(value, pd.Timestamp):
            return np.datetime64(value, unit)
        else:
            # Try to convert as timestamp
            return np.datetime64(value, unit)
    
    def _process_pandas_dataframe(self, df):
        """Process pandas DataFrame and extract columns.
        
        Args:
            df: pandas DataFrame with OHLCV columns (case-insensitive matching)
        """
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
        """Get data series by key using bracket notation.
        
        Args:
            key: Series name (e.g., 'open', 'close', 'volume')
            
        Returns:
            Value from internal dictionary
            
        Raises:
            PyTAExceptionDataSeriesNonFound: If series not found
        """
        if key not in self._data:
            raise PyTAExceptionDataSeriesNonFound(key)
        return self._data[key]
    
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

