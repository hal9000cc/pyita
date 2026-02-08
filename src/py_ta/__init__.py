"""py-ta: Technical analysis library for stock quotes.

This module provides technical indicators for analyzing stock market data.
Indicators are loaded lazily when first accessed.

Example:
    >>> import py_ta as ta
    >>> quotes = ta.Quotes(open, high, low, close)
    >>> bb = ta.bollinger_bands(quotes, period=20)
    >>> sma = ta.sma(quotes, period=20)
"""
import importlib
import re
from pathlib import Path

from .quotes import Quotes
from .exceptions import (
    PyTAException,
    PyTAExceptionIndicatorNotFound,
    PyTAExceptionBadParameterValue,
    PyTAExceptionBadSeriesData,
    PyTAExceptionDataSeriesNonFound,
)


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    # Try to get version from installed package metadata
    try:
        from importlib.metadata import version
        return version('py-ta')
    except Exception:
        pass
    
    # Fallback: read from pyproject.toml (for development)
    try:
        pyproject_path = Path(__file__).parent.parent.parent / 'pyproject.toml'
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding='utf-8')
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    # Final fallback
    return "1.0.0"


__version__ = _get_version()
__all__ = [
    'Quotes',
    'PyTAException',
    'PyTAExceptionIndicatorNotFound',
    'PyTAExceptionBadParameterValue',
    'PyTAExceptionBadSeriesData',
    'PyTAExceptionDataSeriesNonFound',
]

# Cache for lazy-loaded indicators
_indicator_cache = {}


def __getattr__(name):
    """Lazy loading of indicators.
    
    When an indicator is accessed (e.g., ta.bollinger_bands), this function:
    1. Checks if it's already in the cache
    2. If not, tries to import from indicators/{name}.py
    3. Caches and returns the get_indicator_out function
    
    Args:
        name: Name of the indicator (e.g., 'bollinger_bands', 'sma', 'ema')
        
    Returns:
        The get_indicator_out function from the indicator module
        
    Raises:
        PyTAExceptionIndicatorNotFound: If the indicator module or function is not found
    """

    if name in ('__bases__', '__test__'):
        return None

    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _indicator_cache:
        return _indicator_cache[name]
    
    try:
        module = importlib.import_module(f'.indicators.{name}', __package__)
        func = module.get_indicator_out
        _indicator_cache[name] = func
        return func
    except (ImportError, AttributeError) as e:
        raise PyTAExceptionIndicatorNotFound(name) from e


def __dir__():
    """List available attributes including cached indicators."""
    base_attrs = ['Quotes', '__version__']
    return sorted(base_attrs + list(_indicator_cache.keys()))

