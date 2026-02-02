"""Custom exceptions for py-ta library."""


class PyTAException(Exception):
    """Base exception class for all py-ta errors."""
    pass


class PyTAExceptionIndicatorNotFound(PyTAException):
    """Raised when an indicator is not found.
    
    This exception is raised when trying to access an indicator that doesn't exist
    in the indicators directory or cannot be imported.
    
    Attributes:
        indicator_name: Name of the indicator that was not found
    """
    
    def __init__(self, indicator_name):
        """Initialize the exception.
        
        Args:
            indicator_name: Name of the indicator that was not found
        """
        self.indicator_name = indicator_name
        super().__init__(f'Indicator "{self.indicator_name}" not found.')


class PyTAExceptionBadParameterValue(PyTAException):
    """Raised when a parameter has an invalid value.
    
    This exception is raised when an indicator or function receives a parameter
    with an invalid value (e.g., negative period, invalid type, etc.).
    
    Attributes:
        reason: Description of why the parameter value is invalid
    """
    
    def __init__(self, reason):
        """Initialize the exception.
        
        Args:
            reason: Description of why the parameter value is invalid
        """
        self.reason = reason
        super().__init__(f'Bad parameter value: {reason}')


class PyTAExceptionBadSeriesData(PyTAException):
    """Raised when series data is invalid or malformed.
    
    This exception is raised when DataSeries initialization fails due to invalid
    or incompatible data (e.g., missing required columns, incompatible array lengths, etc.).
    
    Attributes:
        reason: Description of why the series data is invalid
    """
    
    def __init__(self, reason):
        """Initialize the exception.
        
        Args:
            reason: Description of why the series data is invalid
        """
        self.reason = reason
        super().__init__(f'Bad series data: {reason}')


class PyTAExceptionTooLittleData(PyTAException):
    """Raised when there is insufficient data for calculation.
    
    This exception is raised when an indicator or function requires more data
    than is available (e.g., period > data length).
    
    Attributes:
        reason: Description of why there is insufficient data
    """
    
    def __init__(self, reason):
        """Initialize the exception.
        
        Args:
            reason: Description of why there is insufficient data
        """
        self.reason = reason
        super().__init__(f'Too little data: {reason}')

