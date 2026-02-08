# py-ta

Fast and simple technical analysis library for Python.

## Features

- **Pure native Python** - written in clean Python, easy to read, understand, and modify
- **Blazingly fast** - uses NumPy vectorized operations and Numba JIT compilation for maximum performance
- **Simple API** - intuitive interface for working with OHLCV data
- **Rich set of indicators** - 28 popular technical indicators
- **Flexible** - easy to create custom indicators
- **Independent** - no external data sources required, bring your own data from any source
- **Compatible** - supports Python 3.9+ (tested up to 3.14)
- **Integrations** - works with NumPy arrays, pandas DataFrames, and CCXT

## Installation

From PyPI:

```bash
pip install py-ta
```

From source:

```bash
git clone https://github.com/hal9000cc/py_ta.git
cd py_ta
pip install -e .
```

## Quick Start

**From NumPy arrays:**

```python
import py_ta as ta
import numpy as np

# Create quotes from arrays
quotes = ta.Quotes(
    open=np.array([100.0, 102.0, 101.0, 103.0]),
    high=np.array([105.0, 106.0, 104.0, 107.0]),
    low=np.array([99.0, 101.0, 100.0, 102.0]),
    close=np.array([102.0, 103.0, 101.0, 105.0])
)

# Calculate indicator
sma = ta.sma(quotes, period=3, value='close')
print(sma.sma)  # [nan, nan, 102.0, 103.0]
```

**From pandas DataFrame:**

```python
import pandas as pd
import py_ta as ta

# Load data from CSV
df = pd.read_csv('data.csv')
quotes = ta.Quotes(df)

# Calculate Bollinger Bands
bb = ta.bollinger_bands(quotes, period=20, deviation=2)
print(bb.up_line)    # Upper band
print(bb.mid_line)   # Middle line
print(bb.down_line)  # Lower band
```

**From CCXT (cryptocurrency exchanges):**

```python
import ccxt
import py_ta as ta

# Fetch data from exchange
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
quotes = ta.Quotes(ohlcv)

# Calculate RSI
rsi = ta.rsi(quotes, period=14)
print(rsi.rsi)  # Values from 0 to 100
```

## Creating Quotes Object

The `Quotes` object encapsulates OHLCV data and can be created from various sources.

**Minimal variant (OHLC only):**

```python
import numpy as np
import py_ta as ta

quotes = ta.Quotes(
    open=np.array([100.0, 102.0, 101.0]),
    high=np.array([105.0, 106.0, 104.0]),
    low=np.array([99.0, 101.0, 100.0]),
    close=np.array([102.0, 103.0, 101.0])
)
```

**With volume:**

```python
quotes = ta.Quotes(
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=np.array([1000, 1200, 900])
)
```

**With volume and time:**

```python
quotes = ta.Quotes(
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=volume,
    time=np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[ms]')
)
```

**From pandas DataFrame:**

```python
import pandas as pd

df = pd.DataFrame({
    'open': [100.0, 102.0],
    'high': [105.0, 106.0],
    'low': [99.0, 101.0],
    'close': [102.0, 103.0],
    'volume': [1000, 1200]
})

quotes = ta.Quotes(df)
```

**From dictionary:**

```python
data = {
    'open': [100.0, 102.0],
    'high': [105.0, 106.0],
    'low': [99.0, 101.0],
    'close': [102.0, 103.0]
}

quotes = ta.Quotes(data)
```

**From list of lists (CCXT format):**

```python
# CCXT returns: [[timestamp, open, high, low, close, volume], ...]
ohlcv = [
    [1609459200000, 100.0, 105.0, 99.0, 102.0, 1000],
    [1609545600000, 102.0, 106.0, 101.0, 103.0, 1200]
]

quotes = ta.Quotes(ohlcv)
```

## Available Indicators

### Moving Averages

- **`sma(quotes, period, value='close')`** - Simple Moving Average
  - Requires: OHLC
  - Returns: `sma`

- **`ema(quotes, period, value='close')`** - Exponential Moving Average
  - Requires: OHLC
  - Returns: `ema`

- **`tema(quotes, period, value='close')`** - Triple Exponential Moving Average
  - Requires: OHLC
  - Returns: `tema`

- **`vwma(quotes, period, value='close')`** - Volume Weighted Moving Average
  - Requires: OHLC + volume
  - Returns: `vwma`

- **`ma(quotes, period, value='close', ma_type='sma')`** - Moving Average (universal)
  - Requires: OHLC
  - Returns: `ma`
  - See [Moving Average Types](#moving-average-types) for available `ma_type` options

### Trend Indicators

- **`adx(quotes, period=14, smooth=14, ma_type='mma')`** - Average Directional Index
  - Requires: OHLC
  - Returns: `adx`, `p_di` (Plus DI), `m_di` (Minus DI)

- **`aroon(quotes, period=14)`** - Aroon Indicator
  - Requires: OHLC (uses high, low)
  - Returns: `up`, `down`, `oscillator`

- **`parabolic_sar(quotes, start=0.02, maximum=0.2, increment=0.02)`** - Parabolic SAR
  - Requires: OHLC
  - Returns: `sar`, `signal`

- **`supertrend(quotes, period=10, multiplier=3, ma_type='mma')`** - SuperTrend
  - Requires: OHLC
  - Returns: `supertrend`, `signal`

- **`macd(quotes, period_fast=12, period_slow=26, period_signal=9, value='close')`** - Moving Average Convergence Divergence
  - Requires: OHLC
  - Returns: `macd`, `signal`, `histogram`

- **`ichimoku(quotes, period_short=9, period_mid=26, period_long=52, offset_senkou=26, offset_chikou=26)`** - Ichimoku Cloud
  - Requires: OHLC (uses high, low)
  - Returns: `tenkan`, `kijun`, `senkou_a`, `senkou_b`, `chikou`

### Oscillators

- **`rsi(quotes, period=14, ma_type='mma', value='close')`** - Relative Strength Index
  - Requires: OHLC
  - Returns: `rsi` (values from 0 to 100)

- **`stochastic(quotes, period=5, period_d=3, smooth=3, ma_type='sma')`** - Stochastic Oscillator
  - Requires: OHLC
  - Returns: `value_k`, `value_d`, `oscillator`

- **`williams_r(quotes, period=14)`** - Williams %R
  - Requires: OHLC
  - Returns: `williams_r` (values from -100 to 0)

- **`cci(quotes, period=20)`** - Commodity Channel Index
  - Requires: OHLC
  - Returns: `cci`

- **`mfi(quotes, period=14)`** - Money Flow Index
  - Requires: OHLC + volume
  - Returns: `mfi`

- **`roc(quotes, period=14, value='close')`** - Rate of Change
  - Requires: OHLC
  - Returns: `roc`

- **`awesome(quotes, period_fast=5, period_slow=34, normalized=False)`** - Awesome Oscillator
  - Requires: OHLC (uses high, low)
  - Returns: `awesome`

- **`trix(quotes, period, value='close')`** - Triple Exponential Average Oscillator
  - Requires: OHLC
  - Returns: `trix`

### Volatility

- **`bollinger_bands(quotes, period=20, deviation=2, ma_type='sma', value='close')`** - Bollinger Bands
  - Requires: OHLC
  - Returns: `mid_line`, `up_line`, `down_line`, `width`, `z_score`

- **`atr(quotes, smooth=14, ma_type='mma')`** - Average True Range
  - Requires: OHLC
  - Returns: `atr`, `atrp` (percentage ATR), `tr` (True Range)

- **`keltner(quotes, period=10, multiplier=1, period_atr=10, ma_type='ema')`** - Keltner Channels
  - Requires: OHLC
  - Returns: `mid_line`, `up_line`, `down_line`, `width`

- **`chandelier(quotes, period=22, multiplier=3, use_close=False)`** - Chandelier Exit
  - Requires: OHLC
  - Returns: `exit_long`, `exit_short`

### Volume Indicators

- **`obv(quotes)`** - On-Balance Volume
  - Requires: OHLC + volume (uses close, volume)
  - Returns: `obv`

- **`vwap(quotes)`** - Volume Weighted Average Price
  - Requires: OHLC + volume
  - Returns: `vwap`

- **`volume_osc(quotes, period_short=5, period_long=10, ma_type='ema')`** - Volume Oscillator
  - Requires: volume
  - Returns: `osc` (percentage difference between short and long MA of volume)

- **`adl(quotes, ma_period=None, ma_type='sma')`** - Accumulation/Distribution Line
  - Requires: OHLC + volume
  - Returns: `adl`, `adl_ema` (if ma_period is specified)

### Other Indicators

- **`zigzag(quotes, delta=0.02, depth=1, type='high_low', end_points=False)`** - ZigZag
  - Requires: OHLC
  - Returns: `pivots`, `pivot_types` (1 = High, -1 = Low, 0 = no pivot)

## Moving Average Types

Many indicators support selecting the moving average type via the `ma_type` parameter:

- **`sma`** - Simple Moving Average
  - All values have equal weight
  - Initialization: SMA over first `period` values

- **`ema`** - Exponential Moving Average
  - More weight given to recent values
  - Smoothing coefficient: α = 2 / (period + 1)
  - Initialization: SMA over first `period` values

- **`mma`** (or `smma`, `rma`) - Modified/Smoothed Moving Average
  - Similar to EMA but with slower response
  - Smoothing coefficient: α = 1 / period
  - Initialization: SMA over first `period` values
  - Used in RSI, ADX indicators

- **`ema0`** - EMA with first value initialization
  - No warmup period required

- **`mma0`** - MMA with first value initialization
  - No warmup period required

- **`emaw`** - EMA with dynamic warmup period (TA-Lib compatible)

- **`mmaw`** - MMA with dynamic warmup period (TA-Lib compatible)

Essentially, technical analysis uses only two moving average algorithms - simple (sma) and exponential (ema). SMA is always calculated the same way. EMA is also calculated consistently, but there can be variations in how it uses the period (smoothing coefficient calculation) and how it's initialized. This is what distinguishes the different ema types supported by py-ta.

**Usage example:**

```python
# Bollinger Bands with EMA
bb = ta.bollinger_bands(quotes, period=20, deviation=2, ma_type='ema')

# Keltner Channels with MMA
keltner = ta.keltner(quotes, period=20, multiplier=2, period_atr=10, ma_type='mma')

# RSI uses MMA by default (can be changed)
rsi = ta.rsi(quotes, period=14, ma_type='ema')
```

## System Requirements

- **Python**: 3.9+ (tested up to 3.14)
- **Required dependencies**:
  - numpy >= 1.20.0
  - numba >= 0.53.0
- **Optional dependencies**:
  - pandas >= 1.3.0 (for DataFrame support)

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT License

Copyright (c) 2022 Aleksandr Kuznetsov hal@hal9000.cc

## Contact

- **GitHub Issues**: https://github.com/hal9000cc/py_ta/issues
- **Email**: hal@hal9000.cc
