#!/usr/bin/env python3
"""Script for downloading test OHLCV data from cryptocurrency exchanges via CCXT.

This script downloads historical OHLCV data and saves it as a pickle file
for use in testing py-ta library.

Usage:
    python download_test_data.py [--source SOURCE] [--symbol SYMBOL] 
                                  [--timeframe TIMEFRAME] [--year YEAR]

Example:
    python download_test_data.py --source binance --symbol BTC/USDT --timeframe 1h --year 2024
"""
import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ccxt
from tqdm import tqdm


def get_previous_year():
    """Get the previous year (full year).
    
    Returns:
        int: Previous year (current_year - 1)
    """
    return datetime.now().year - 1


def get_year_timestamps(year):
    """Get start and end timestamps for a given year.
    
    Args:
        year: Year (e.g., 2024)
        
    Returns:
        tuple: (start_timestamp_ms, end_timestamp_ms)
            - start: January 1, 00:00:00 UTC
            - end: Current date minus one day, 23:59:59 UTC
    """
    # Start: January 1, 00:00:00 UTC
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    
    # End: Current date minus one day, 23:59:59 UTC
    end_date = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999000)
    end_date = end_date.replace(day=end_date.day - 1)
    end_ms = int(end_date.timestamp() * 1000)
    
    return start_ms, end_ms


def fetch_ohlcv_batch(exchange, symbol, timeframe, since, limit=500):
    """Fetch a batch of OHLCV data.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h')
        since: Start timestamp in milliseconds
        limit: Maximum number of bars to fetch (default: 500)
        
    Returns:
        list: List of OHLCV bars [[timestamp, open, high, low, close, volume], ...]
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        raise


def download_ohlcv_data(exchange, symbol, timeframe, start_ms, end_ms, limit=500):
    """Download OHLCV data in batches.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair
        timeframe: Timeframe
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        limit: Number of bars per batch (default: 500)
        
    Returns:
        list: Complete list of OHLCV bars
    """
    all_bars = []
    current_since = start_ms
    
    # Estimate total number of bars for progress bar
    # Approximate: (end_ms - start_ms) / (timeframe_ms * limit)
    timeframe_hours = {'1h': 1, '4h': 4, '1d': 24, '1w': 168}.get(timeframe, 1)
    timeframe_ms = timeframe_hours * 60 * 60 * 1000
    estimated_batches = max(1, int((end_ms - start_ms) / (timeframe_ms * limit)) + 1)
    
    print(f"Starting download...")
    print(f"  Source: {exchange.id}")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Date range: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}")
    print()
    
    with tqdm(total=estimated_batches, desc="Downloading", unit="batch") as pbar:
        while current_since < end_ms:
            batch = fetch_ohlcv_batch(exchange, symbol, timeframe, current_since, limit)
            
            if not batch:
                break
            
            all_bars.extend(batch)
            
            # Update progress
            pbar.update(1)
            
            # Check if we've reached the end
            last_timestamp = batch[-1][0]
            if last_timestamp >= end_ms:
                break
            
            # Next batch starts from the last timestamp + 1ms
            current_since = last_timestamp + 1
            
            # Safety check: if we got fewer bars than requested, we're done
            if len(batch) < limit:
                break
    
    return all_bars


def generate_filename(source, symbol, timeframe, year):
    """Generate filename for saved data.
    
    Args:
        source: Exchange name (e.g., 'binance')
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h')
        year: Year (e.g., 2024)
        
    Returns:
        str: Filename (e.g., 'binance_BTC_USDT_1h_2024.pkl')
    """
    # Convert symbol to uppercase and replace / with _
    symbol_clean = symbol.upper().replace('/', '_')
    source_clean = source.upper()
    
    return f"{source_clean}_{symbol_clean}_{timeframe}_{year}.pkl"


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Download test OHLCV data from cryptocurrency exchanges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_test_data.py
  python download_test_data.py --source binance --symbol BTC/USDT --timeframe 1h --year 2024
  python download_test_data.py --symbol ETH/USDT --timeframe 4h
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe (default: 1h)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to download (default: previous year)'
    )
    
    return parser.parse_args()


def initialize_exchange(source):
    """Initialize CCXT exchange instance.
    
    Args:
        source: Exchange name (e.g., 'binance')
        
    Returns:
        ccxt.Exchange: Initialized exchange instance
        
    Raises:
        SystemExit: If exchange cannot be initialized
    """
    try:
        exchange_class = getattr(ccxt, source)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        print(f"Initialized exchange: {exchange.id}")
        return exchange
    except AttributeError:
        print(f"Error: Exchange '{source}' not found in CCXT", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing exchange: {e}", file=sys.stderr)
        sys.exit(1)


def save_data(bars, source, symbol, timeframe, year):
    """Save downloaded data to pickle file.
    
    Converts raw OHLCV data to numpy arrays with proper types:
    - time: datetime64[ms] from timestamps
    - open, high, low, close: float arrays
    - volume: float array
    
    Args:
        bars: List of OHLCV bars [[timestamp_ms, open, high, low, close, volume], ...]
        source: Exchange name
        symbol: Trading pair
        timeframe: Timeframe
        year: Year
        
    Raises:
        SystemExit: If data cannot be saved
    """
    # Convert to numpy arrays
    # bars format: [[timestamp_ms, open, high, low, close, volume], ...]
    timestamps_ms = np.array([bar[0] for bar in bars], dtype=np.int64)
    time = timestamps_ms.astype('datetime64[ms]')
    
    open_data = np.array([bar[1] for bar in bars], dtype=float)
    high_data = np.array([bar[2] for bar in bars], dtype=float)
    low_data = np.array([bar[3] for bar in bars], dtype=float)
    close_data = np.array([bar[4] for bar in bars], dtype=float)
    volume_data = np.array([bar[5] for bar in bars], dtype=float)
    
    # Create dictionary with numpy arrays
    data_dict = {
        'time': time,
        'open': open_data,
        'high': high_data,
        'low': low_data,
        'close': close_data,
        'volume': volume_data,
    }
    
    # Generate filename
    filename = generate_filename(source, symbol, timeframe, year)
    
    # Get script directory for saving
    script_dir = Path(__file__).parent
    filepath = script_dir / filename
    
    # Save data
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        file_size = filepath.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print()
        print("=" * 60)
        print("Download completed successfully!")
        print(f"  Total bars: {len(bars):,}")
        print(f"  Saved to: {filepath}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error saving data: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Determine year
    if args.year is None:
        args.year = get_previous_year()
    
    # Initialize exchange
    exchange = initialize_exchange(args.source)
    
    # Get timestamps
    start_ms, end_ms = get_year_timestamps(args.year)
    
    # Download data
    try:
        bars = download_ohlcv_data(
            exchange,
            args.symbol,
            args.timeframe,
            start_ms,
            end_ms,
            limit=500
        )
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not bars:
        print("Error: No data downloaded", file=sys.stderr)
        sys.exit(1)
    
    # Save data
    save_data(bars, args.source, args.symbol, args.timeframe, args.year)


if __name__ == '__main__':
    main()

