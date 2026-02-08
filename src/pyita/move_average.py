import numpy as np
from numba import njit
from enum import Enum
from .exceptions import PyTAExceptionTooLittleData


class MA_Type(Enum):

    ema = 1
    sma = 2
    mma = 3
    ema0 = 4
    mma0 = 5
    ema_warmup = 6
    mma_warmup = 7

    @staticmethod
    def cast(str_value):

        if str_value == 'ema':  # Classic EMA with initialization via SMA (alpha = 2.0 / (period + 1))
            return MA_Type.ema
        if str_value == 'sma':  # Classic SMA with initialization via SMA
            return MA_Type.sma
        if str_value == 'mma':  # Modified EMA with initialization via SMA (alpha = 1.0 / period)
            return MA_Type.mma
        if str_value == 'ema0':  # Classic EMA with initialization via first data element
            return MA_Type.ema0
        if str_value == 'mma0':  # Modified EMA with initialization via first data element (alpha = 1.0 / period)
            return MA_Type.mma0
        if str_value == 'emaw':  # EMA with dynamic-alpha warm-up (TA-Lib compatible)
            return MA_Type.ema_warmup
        if str_value == 'mmaw':  # MMA (SMMA) with dynamic-alpha warm-up (TA-Lib compatible)
            return MA_Type.mma_warmup

        raise ValueError(f'Unknown move average type: {str_value}')


@njit(cache=True)
def get_first_index_not_nan(values):

    for i, value in enumerate(values):
        if not np.isnan(value):
            return i

    return len(values)


@njit(cache=True)
def ema_calculate(source_values, alpha, first_value=np.nan, start=0):

    alpha_n = 1.0 - alpha

    if np.isnan(first_value):
        # skip first nans
        for i, ema_value in enumerate(source_values):
            if not np.isnan(ema_value):
                start = i
                break
        else:
            ema_value = source_values[start]
    else:
        ema_value = first_value

    result = np.empty(len(source_values), dtype=float)
    result[: start] = np.nan
    result[start] = ema_value

    for i in range(start + 1, len(source_values)):
        ema_value = source_values[i] * alpha + ema_value * alpha_n
        result[i] = ema_value

    return result


def sma_calculate(source_values, period):

    if period == 1:
        return source_values

    data_len = len(source_values)
    if data_len < period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {period}')

    weights = np.ones(period, dtype=source_values.dtype) / period

    out = np.convolve(source_values, weights)[:-period+1]
    out[:period - 1] = np.nan

    return out


def iema_calculate(source_values, period, alpha):

    start = get_first_index_not_nan(source_values)

    data_len = len(source_values)
    if data_len < start + period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {start + period}')

    first_value = source_values[start: start + period].sum() / period
    return ema_calculate(source_values, alpha, first_value, start + period - 1)


@njit(cache=True)
def ema_warmup_init(source_values, period, start):

    init_ema = source_values[start]

    for i in range(1, period):
        k = 1.0 / (i + 1)
        init_ema = source_values[start + i] * k + init_ema * (1.0 - k)

    return init_ema


def ema_warmup_calculate(source_values, period, alpha):

    start = get_first_index_not_nan(source_values)
    if start >= len(source_values):
        result = np.empty_like(source_values)
        result[:] = np.nan
        return result

    data_len = len(source_values)
    if data_len < start + period:
        raise PyTAExceptionTooLittleData(f'data length {data_len} < {start + period}')

    prev_ema = ema_warmup_init(source_values, period, start)

    result = ema_calculate(source_values, alpha, prev_ema, start + period - 1)

    return result


def ma_calculate(source_values, period, ma_type):

    if ma_type == MA_Type.sma:
        return sma_calculate(source_values, period)
    if ma_type == MA_Type.ema0:
        alpha = 2.0 / (period + 1)
        return ema_calculate(source_values, alpha)
    if ma_type == MA_Type.mma0:
        alpha = 1.0 / period
        return ema_calculate(source_values, alpha)
    if ma_type == MA_Type.ema:
        alpha = 2.0 / (period + 1)
        return iema_calculate(source_values, period, alpha)
    if ma_type == MA_Type.mma:
        alpha = 1.0 / period
        return iema_calculate(source_values, period, alpha)
    if ma_type == MA_Type.ema_warmup:
        alpha = 2.0 / (period + 1)
        return ema_warmup_calculate(source_values, period, alpha)
    if ma_type == MA_Type.mma_warmup:
        alpha = 1.0 / period
        return ema_warmup_calculate(source_values, period, alpha)

    raise ValueError(f'Bad ma_type value: {ma_type}')
