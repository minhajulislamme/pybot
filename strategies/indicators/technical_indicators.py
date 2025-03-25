"""
Technical indicators module for trading strategies.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return lower_band, middle_band, upper_band

    @staticmethod
    def calculate_moving_average(data, period, ma_type='sma'):
        if ma_type.lower() == 'sma':
            return data.rolling(window=period).mean()
        elif ma_type.lower() == 'ema':
            return data.ewm(span=period, adjust=False).mean()
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")

    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def calculate_stochastic(data, k_period=14, d_period=3):
        low_min = data.rolling(k_period).min()
        high_max = data.rolling(k_period).max()
        k = 100 * ((data - low_min) / (high_max - low_min))
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def calculate_atr(data, period=14):
        high = data.rolling(window=period).max()
        low = data.rolling(window=period).min()
        close = data.shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()