import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Base strategy class that all strategies should inherit from"""
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} on {timeframe} timeframe")
    
    @abstractmethod
    def calculate_signals(self, data):
        """Calculate buy/sell signals based on the strategy logic"""
        pass
    
    def prepare_data(self, klines):
        """Convert raw klines data to a pandas DataFrame"""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                          'close_time', 'quote_asset_volume', 'number_of_trades',
                                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convert string values to numeric
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df


class SMACrossoverStrategy(Strategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, symbol, timeframe, short_period=9, long_period=21):
        super().__init__(symbol, timeframe)
        self.short_period = short_period
        self.long_period = long_period
        logger.info(f"SMA Crossover: short_period={short_period}, long_period={long_period}")
    
    def calculate_signals(self, data):
        """Calculate buy/sell signals based on SMA crossover"""
        df = data.copy()
        
        # Calculate short and long SMAs
        df['short_sma'] = df['close'].rolling(window=self.short_period).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_period).mean()
        
        # Generate signals
        df['signal'] = 0
        
        # Crossover signals
        df['prev_short_sma'] = df['short_sma'].shift(1)
        df['prev_long_sma'] = df['long_sma'].shift(1)
        
        # Buy signal: short SMA crosses above long SMA
        buy_signal = (df['prev_short_sma'] < df['prev_long_sma']) & (df['short_sma'] > df['long_sma'])
        df.loc[buy_signal, 'signal'] = 1
        
        # Sell signal: short SMA crosses below long SMA
        sell_signal = (df['prev_short_sma'] > df['prev_long_sma']) & (df['short_sma'] < df['long_sma'])
        df.loc[sell_signal, 'signal'] = -1
        
        return df


class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) Strategy"""
    
    def __init__(self, symbol, timeframe, period=14, overbought=70, oversold=30):
        super().__init__(symbol, timeframe)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        logger.info(f"RSI Strategy: period={period}, overbought={overbought}, oversold={oversold}")
    
    def calculate_rsi(self, data):
        """Calculate RSI indicator"""
        df = data.copy()
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_signals(self, data):
        """Calculate buy/sell signals based on RSI values"""
        df = self.calculate_rsi(data)
        
        # Generate signals
        df['signal'] = 0
        
        # RSI crosses below oversold threshold
        df['prev_rsi'] = df['rsi'].shift(1)
        
        # Buy signal: RSI crosses from below to above oversold level
        buy_signal = (df['prev_rsi'] < self.oversold) & (df['rsi'] > self.oversold)
        df.loc[buy_signal, 'signal'] = 1
        
        # Sell signal: RSI crosses from above to below overbought level
        sell_signal = (df['prev_rsi'] > self.overbought) & (df['rsi'] < self.overbought)
        df.loc[sell_signal, 'signal'] = -1
        
        return df


class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) Strategy"""
    
    def __init__(self, symbol, timeframe, fast_period=12, slow_period=26, signal_period=9):
        super().__init__(symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        logger.info(f"MACD Strategy: fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
    
    def calculate_macd(self, data):
        """Calculate MACD indicator"""
        df = data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_signals(self, data):
        """Calculate buy/sell signals based on MACD indicator"""
        df = self.calculate_macd(data)
        
        # Generate signals
        df['signal'] = 0
        
        # Calculate previous values
        df['prev_macd'] = df['macd'].shift(1)
        df['prev_macd_signal'] = df['macd_signal'].shift(1)
        
        # Buy signal: MACD line crosses above signal line
        buy_signal = (df['prev_macd'] < df['prev_macd_signal']) & (df['macd'] > df['macd_signal'])
        df.loc[buy_signal, 'signal'] = 1
        
        # Sell signal: MACD line crosses below signal line
        sell_signal = (df['prev_macd'] > df['prev_macd_signal']) & (df['macd'] < df['macd_signal'])
        df.loc[sell_signal, 'signal'] = -1
        
        return df


def create_strategy(strategy_name, symbol, timeframe, **kwargs):
    """Factory function to create the appropriate strategy"""
    strategies = {
        'SMA_CROSSOVER': SMACrossoverStrategy,
        'RSI': RSIStrategy,
        'MACD': MACDStrategy
    }
    
    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        logger.error(f"Unknown strategy: {strategy_name}")
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class(symbol, timeframe, **kwargs)