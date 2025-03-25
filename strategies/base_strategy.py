"""
Base strategy class that all trading strategies will inherit from.
"""

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, client, symbol, timeframe='1h', risk_manager=None):
        """
        Initialize the base strategy.
        
        Args:
            client: BinanceClient instance
            symbol (str): Trading symbol
            timeframe (str): Candlestick timeframe
            risk_manager: RiskManager instance
        """
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_manager = risk_manager
        self.df = None
        self.last_signal = None
        self.signal_time = None
        self.indicators = {}  # Store calculated indicators
    
    def prepare_data(self):
        """Prepare data for strategy calculation"""
        try:
            # Get historical data
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=200  # Increased to ensure enough data for indicators
            )
            
            # Check if we got valid data
            if klines is None or isinstance(klines, pd.DataFrame) and klines.empty:
                logger.error("Failed to fetch historical data")
                return False
                
            self.df = klines
            
            # Use a small synthetic dataset for testing if the API call failed
            if self.df is None:
                logger.warning("Using synthetic data for testing")
                dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
                self.df = pd.DataFrame({
                    'open': np.linspace(10000, 15000, 200) + np.random.normal(0, 100, 200),
                    'high': np.linspace(10000, 15000, 200) + np.random.normal(0, 200, 200),
                    'low': np.linspace(10000, 15000, 200) + np.random.normal(0, 200, 200),
                    'close': np.linspace(10000, 15000, 200) + np.random.normal(0, 100, 200),
                    'volume': np.random.normal(100, 10, 200)
                }, index=dates)
                
                # Ensure high is actually the highest and low is the lowest
                for i in range(len(self.df)):
                    self.df.loc[self.df.index[i], 'high'] = max(
                        self.df.loc[self.df.index[i], 'open'],
                        self.df.loc[self.df.index[i], 'close'],
                        self.df.loc[self.df.index[i], 'high']
                    )
                    self.df.loc[self.df.index[i], 'low'] = min(
                        self.df.loc[self.df.index[i], 'open'],
                        self.df.loc[self.df.index[i], 'close'],
                        self.df.loc[self.df.index[i], 'low']
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    @abstractmethod
    def calculate_indicators(self):
        """
        Calculate technical indicators required for the strategy.
        This method should be implemented by each strategy subclass.
        """
        pass
    
    @abstractmethod
    def generate_signal(self):
        """
        Generate trading signals based on indicators.
        This method should be implemented by each strategy subclass.
        
        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for neutral)
        """
        pass
    
    def update_data(self, kline_data=None):
        """
        Update the data with the latest kline.
        
        Args:
            kline_data (dict, optional): Latest kline data from websocket
        """
        if kline_data:
            # Extract kline data
            timestamp = datetime.fromtimestamp(kline_data['t'] / 1000)
            new_row = {
                'open': float(kline_data['o']),
                'high': float(kline_data['h']),
                'low': float(kline_data['l']),
                'close': float(kline_data['c']),
                'volume': float(kline_data['v']),
                'close_time': datetime.fromtimestamp(kline_data['T'] / 1000),
                'quote_volume': float(kline_data['q']),
                'trades': int(kline_data['n']),
                'taker_buy_base': float(kline_data['V']),
                'taker_buy_quote': float(kline_data['Q']),
                'ignore': 0
            }
            
            # If df is empty, fetch data first
            if self.df is None or len(self.df) == 0:
                self.prepare_data()
                return
            
            # If the timestamp exists, update that row
            if timestamp in self.df.index:
                for col, value in new_row.items():
                    self.df.loc[timestamp, col] = value
            else:
                # Create a new row
                new_df = pd.DataFrame([new_row], index=[timestamp])
                self.df = pd.concat([self.df, new_df])
                
            # Keep only the last 300 rows to avoid memory issues
            if len(self.df) > 300:
                self.df = self.df.iloc[-300:]
        else:
            # If no kline data is provided, fetch the latest data
            self.prepare_data()
    
    def execute_signal(self, signal):
        """
        Execute a trading signal if it's different from the previous signal.
        
        Args:
            signal (int): Trading signal (1 for buy, -1 for sell, 0 for neutral)
            
        Returns:
            dict: Result of the execution or None if no action was taken
        """
        current_time = datetime.now()
        
        # Skip if the signal is the same as the last one
        if signal == self.last_signal:
            return None
        
        # Check if enough time has passed since the last signal
        if self.signal_time and (current_time - self.signal_time).seconds < 300:  # 5 minutes
            logger.info(f"Skipping signal, too soon after last signal")
            return None
        
        # Update last signal and time
        self.last_signal = signal
        self.signal_time = current_time
        
        # Execute the signal if risk management allows
        if self.risk_manager:
            result = self.risk_manager.execute_trade(self.symbol, signal)
        else:
            logger.warning("No risk manager attached, cannot execute trade")
            result = None
            
        return result
    
    def backtest(self, start_date=None, end_date=None):
        """
        Backtest the strategy on historical data.
        
        Args:
            start_date (str, optional): Start date for backtesting 'YYYY-MM-DD'
            end_date (str, optional): End date for backtesting 'YYYY-MM-DD'
            
        Returns:
            DataFrame: Backtest results
        """
        pass  # This will be implemented in a separate backtesting module