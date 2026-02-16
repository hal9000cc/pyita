"""Helper functions for pyita library."""
from .constants import VALID_PRICE_VALUES, VALID_PRICE_AND_VOLUME_VALUES
from .exceptions import PyTAExceptionBadParameterValue


def validate_value_par(value, allow_volume=False):
    """Validate the value parameter for indicators.
    
    Args:
        value: Value to validate - should be one of 'open', 'high', 'low', 'close', or optionally 'volume'
        allow_volume: If True, allows 'volume' as a valid value (default: False)
        
    Raises:
        PyTAExceptionBadParameterValue: If value is not in the list of valid values
    """
    valid_values = VALID_PRICE_AND_VOLUME_VALUES if allow_volume else VALID_PRICE_VALUES
    if value not in valid_values:
        raise PyTAExceptionBadParameterValue(f'value must be one of {valid_values}, got {value}')

