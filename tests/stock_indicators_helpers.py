"""Helpers for converting data between py_ta and stock-indicators formats."""
import pickle
from pathlib import Path

import numpy as np
from py_ta.indicator_result import IndicatorResult

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


def quotes_to_si(quotes):
    """Convert py_ta Quotes object to list of stock-indicators Quote objects.

    Args:
        quotes: py_ta Quotes object with .open, .high, .low, .close,
                .time, .volume attributes (numpy arrays)

    Returns:
        list[Quote]: List of stock-indicators Quote objects
    """
    return _dict_to_si_quotes({
        'time': quotes.time,
        'open': quotes.open,
        'high': quotes.high,
        'low': quotes.low,
        'close': quotes.close,
        'volume': quotes.volume,
    })


def si_results_to_numpy(results, attrs):
    """Convert stock-indicators results to dict of numpy arrays.

    None values are converted to NaN.

    Args:
        results: stock-indicators IndicatorResults (list of result objects)
        attrs: list of attribute names to extract (e.g. ['atr', 'tr', 'atrp'])

    Returns:
        dict[str, np.ndarray]: Attribute names mapped to float64 numpy arrays.
    """
    n = len(results)
    output = {}

    for attr in attrs:
        arr = np.empty(n, dtype=np.float64)
        for i, r in enumerate(results):
            val = getattr(r, attr)
            arr[i] = np.nan if val is None else float(val)
        output[attr] = arr

    return output


def get_si_ref(quotes_filename, si_func_name, *args):
    """Get cached stock-indicators reference values.

    On first call computes indicator via stock-indicators and caches
    the result as a pickle file. Subsequent calls load from cache.

    Args:
        quotes_filename: Name of quotes pickle file (e.g. 'BINANCE_BTC_USDT_1h_2025.pkl')
        si_func_name: Name of stock-indicators function (e.g. 'get_adx')
        *args: Positional arguments to pass to the indicator function.
               String values for enum types (e.g., 'HIGH_LOW', 'SHORT', 'LONG')
               will be automatically converted to enum objects when generating data.

    Returns:
        IndicatorResult: Object with attribute access to numpy arrays
    """
    cache_path = _build_cache_path(quotes_filename, si_func_name, args)

    # If cache exists, load from cache (no need for stock-indicators)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return IndicatorResult(pickle.load(f))

    # Cache doesn't exist - need to generate it using stock-indicators
    # Convert string enum arguments to enum objects before calling stock-indicators
    converted_args = _convert_args_to_enums(si_func_name, args)
    
    # Import here to avoid requiring stock-indicators as a dependency
    try:
        from stock_indicators import indicators as si
    except ImportError:
        raise ImportError(
            "stock-indicators is required to generate reference data. "
            "Install it with: pip install stock-indicators"
        )

    with open(TEST_DATA_DIR / quotes_filename, 'rb') as f:
        quotes_data = pickle.load(f)

    func = getattr(si, si_func_name)
    si_quotes = _dict_to_si_quotes(quotes_data)
    results = func(si_quotes, *converted_args)
    data_dict = _extract_all_attrs(results)

    with open(cache_path, 'wb') as f:
        pickle.dump(data_dict, f)

    return IndicatorResult(data_dict)


def _format_arg_for_filename(arg):
    """Format argument for use in cache filename.
    
    For enums, uses .name attribute for readability.
    For other types, uses str() representation.
    """
    if hasattr(arg, 'name') and isinstance(arg.name, str):
        return arg.name
    return str(arg)


def _build_cache_path(quotes_filename, si_func_name, args):
    quotes_base = quotes_filename.replace('.pkl', '')
    params_suffix = '-' + ','.join(_format_arg_for_filename(a) for a in args) if args else ''
    cache_name = f'si_ref-{si_func_name}-{quotes_base}{params_suffix}.pkl'
    return TEST_DATA_DIR / cache_name


def _dict_to_si_quotes(data_dict):
    """Convert dictionary to stock-indicators Quote objects.
    
    Imports stock-indicators only when needed.
    """
    from stock_indicators import Quote

    time = data_dict['time']
    open_data = data_dict['open']
    high = data_dict['high']
    low = data_dict['low']
    close = data_dict['close']
    volume = data_dict['volume']

    result = []
    for i in range(len(close)):
        result.append(Quote(
            date=time[i].astype('datetime64[ms]').item(),
            open=float(open_data[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
        ))
    return result


def _convert_args_to_enums(si_func_name, args):
    """Convert string enum arguments to enum objects.
    
    Only converts when needed for data generation. This allows tests to pass
    strings instead of enum objects, avoiding stock-indicators import when cache exists.
    
    Args:
        si_func_name: Name of stock-indicators function
        args: Original arguments (may contain strings for enum types)
        
    Returns:
        tuple: Arguments with enum strings converted to enum objects
    """
    args_list = list(args)
    
    if si_func_name == 'get_zig_zag':
        # First argument after quotes is EndType
        if len(args_list) > 0 and isinstance(args_list[0], str):
            try:
                from stock_indicators.indicators.common.enums import EndType
                args_list[0] = getattr(EndType, args_list[0])
            except (ImportError, AttributeError):
                pass  # If stock-indicators not available, will fail later
    
    elif si_func_name == 'get_chandelier':
        # Last argument is ChandelierType
        if len(args_list) > 0 and isinstance(args_list[-1], str):
            try:
                from stock_indicators.indicators.common.enums import ChandelierType
                args_list[-1] = getattr(ChandelierType, args_list[-1])
            except (ImportError, AttributeError):
                pass  # If stock-indicators not available, will fail later
    
    return tuple(args_list)


def _extract_all_attrs(results):
    """Extract all attributes from stock-indicators results.
    
    Handles both numeric and string values (e.g., point_type in ZigZag).
    """
    result_class = type(results[0])
    attrs = [
        name for name in dir(result_class)
        if isinstance(getattr(result_class, name, None), property)
        and name != 'date' and not name.startswith('_')
    ]

    n = len(results)
    data_dict = {}
    for attr in attrs:
        # Determine dtype by checking first non-None value
        dtype = np.float64
        for r in results:
            val = getattr(r, attr)
            if val is not None:
                if isinstance(val, str):
                    dtype = object
                break
        
        if dtype == object:
            arr = np.empty(n, dtype=object)
            for i, r in enumerate(results):
                val = getattr(r, attr)
                arr[i] = None if val is None else val
        else:
            arr = np.empty(n, dtype=np.float64)
            for i, r in enumerate(results):
                val = getattr(r, attr)
                arr[i] = np.nan if val is None else float(val)
        data_dict[attr] = arr

    return data_dict

