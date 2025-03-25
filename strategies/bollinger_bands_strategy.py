"""
Enhanced Bollinger Bands strategy with volume analysis and trend confirmation.
"""

import logging
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, client, symbol: str, timeframe: str, risk_manager=None,
                 bb_period: int = 20, bb_std: float = 2.0, 
                 volume_factor: float = 2.0, trend_period: int = 200):
        super().__init__(client, symbol, timeframe, risk_manager)
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_factor = volume_factor
        self.trend_period = trend_period
        self.indicators: Dict[str, Any] = {}
        
        # Additional timeframes for confirmation
        self.timeframes = {
            'short': self.get_lower_timeframe(timeframe),
            'primary': timeframe,
            'long': self.get_higher_timeframe(timeframe)
        }
    
    def calculate_indicators(self) -> None:
        """Calculate Bollinger Bands and supporting indicators"""
        if self.df is None or len(self.df) < self.trend_period:
            return
            
        try:
            # Calculate indicators for each timeframe
            for tf_name, tf_data in self.timeframes.items():
                if tf_name == 'primary':
                    data = self.df
                else:
                    data = self.client.get_historical_klines(timeframe=tf_data, limit=300)
                    if data is None or len(data) < self.trend_period:
                        continue
                
                # Calculate Bollinger Bands
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                sma = typical_price.rolling(window=self.bb_period).mean()
                std = typical_price.rolling(window=self.bb_period).std()
                
                self.indicators[f'bb_middle_{tf_name}'] = sma
                self.indicators[f'bb_upper_{tf_name}'] = sma + (std * self.bb_std)
                self.indicators[f'bb_lower_{tf_name}'] = sma - (std * self.bb_std)
                
                # Calculate BB width for squeeze detection
                self.indicators[f'bb_width_{tf_name}'] = (
                    self.indicators[f'bb_upper_{tf_name}'] - 
                    self.indicators[f'bb_lower_{tf_name}']
                ) / self.indicators[f'bb_middle_{tf_name}']
                
                # Calculate trend indicators
                self.indicators[f'sma_{tf_name}'] = data['close'].rolling(window=self.trend_period).mean()
                self.indicators[f'ema_{tf_name}'] = data['close'].ewm(span=self.trend_period).mean()
            
            # Calculate momentum and volatility
            returns = self.df['close'].pct_change()
            self.indicators['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            self.indicators['momentum'] = returns.rolling(window=10).mean() * 100
            
            # Calculate volume indicators
            self.indicators['volume_sma'] = self.df['volume'].rolling(window=20).mean()
            self.indicators['volume_std'] = self.df['volume'].rolling(window=20).std()
            self.indicators['volume_ratio'] = self.df['volume'] / self.indicators['volume_sma']
            
            # Calculate RSI for additional confirmation
            close_diff = self.df['close'].diff()
            gains = close_diff.where(close_diff > 0, 0)
            losses = -close_diff.where(close_diff < 0, 0)
            
            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()
            
            rs = avg_gains / avg_losses
            self.indicators['rsi'] = 100 - (100 / (1 + rs))
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    def check_volume_confirmation(self) -> bool:
        """Check if volume confirms the signal"""
        try:
            current_volume = self.df['volume'].iloc[-1]
            avg_volume = self.indicators['volume_sma'].iloc[-1]
            vol_std = self.indicators['volume_std'].iloc[-1]
            
            return (current_volume > avg_volume * self.volume_factor and
                   current_volume > avg_volume + (2 * vol_std))
        except Exception as e:
            logger.error(f"Error checking volume: {e}")
            return False
    
    def check_bb_squeeze(self) -> bool:
        """Check if Bollinger Bands are in a squeeze"""
        try:
            current_width = self.indicators['bb_width_primary'].iloc[-1]
            avg_width = self.indicators['bb_width_primary'].rolling(window=20).mean().iloc[-1]
            
            return current_width < avg_width * 0.8  # 20% below average width
        except Exception as e:
            logger.error(f"Error checking BB squeeze: {e}")
            return False
    
    def check_trend_alignment(self) -> Optional[int]:
        """Check if price action aligns with multiple timeframe trend"""
        try:
            alignments = []
            
            for tf_name in self.timeframes:
                if (f'bb_middle_{tf_name}' not in self.indicators or
                    f'sma_{tf_name}' not in self.indicators):
                    continue
                
                current_price = self.df['close'].iloc[-1]
                bb_middle = self.indicators[f'bb_middle_{tf_name}'].iloc[-1]
                sma = self.indicators[f'sma_{tf_name}'].iloc[-1]
                
                # Determine trend direction
                if current_price > bb_middle and current_price > sma:
                    alignments.append(1)  # Uptrend
                elif current_price < bb_middle and current_price < sma:
                    alignments.append(-1)  # Downtrend
                else:
                    alignments.append(0)
            
            if not alignments:
                return None
            
            # Return signal if majority agree
            if sum(1 for x in alignments if x > 0) >= len(alignments) / 2:
                return 1
            elif sum(1 for x in alignments if x < 0) >= len(alignments) / 2:
                return -1
            return 0
            
        except Exception as e:
            logger.error(f"Error checking trend alignment: {e}")
            return None
    
    def check_momentum(self) -> Optional[int]:
        """Check if momentum confirms the signal"""
        try:
            momentum = self.indicators['momentum'].iloc[-1]
            volatility = self.indicators['volatility'].iloc[-1]
            
            # Adjust threshold based on volatility
            threshold = 0.1 * (1 + volatility)
            
            if momentum > threshold:
                return 1
            elif momentum < -threshold:
                return -1
            return 0
            
        except Exception as e:
            logger.error(f"Error checking momentum: {e}")
            return None
    
    def check_rsi_confirmation(self, signal: int) -> bool:
        """Check if RSI confirms the trade signal"""
        try:
            rsi = self.indicators['rsi'].iloc[-1]
            
            if signal > 0:  # Buy signal
                return rsi < 50  # Confirm buy when RSI is below midpoint
            elif signal < 0:  # Sell signal
                return rsi > 50  # Confirm sell when RSI is above midpoint
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking RSI confirmation: {e}")
            return False
    
    def get_lower_timeframe(self, timeframe: str) -> str:
        """Get next lower timeframe"""
        tf_map = {'1d': '4h', '4h': '1h', '1h': '15m', '15m': '5m', '5m': '1m'}
        return tf_map.get(timeframe, '15m')
    
    def get_higher_timeframe(self, timeframe: str) -> str:
        """Get next higher timeframe"""
        tf_map = {'1m': '5m', '5m': '15m', '15m': '1h', '1h': '4h', '4h': '1d'}
        return tf_map.get(timeframe, '4h')
    
    def generate_signal(self) -> int:
        """
        Generate trading signal based on enhanced Bollinger Bands strategy.
        Returns:
            int: 1 for buy, -1 for sell, 0 for no action
        """
        if not self.ready_to_trade():
            return 0
            
        try:
            # Get current price and BB values
            current_price = self.df['close'].iloc[-1]
            bb_upper = self.indicators['bb_upper_primary'].iloc[-1]
            bb_lower = self.indicators['bb_lower_primary'].iloc[-1]
            bb_middle = self.indicators['bb_middle_primary'].iloc[-1]
            
            signal = 0
            
            # Check for potential entry points
            if current_price <= bb_lower:
                signal = 1  # Potential buy
            elif current_price >= bb_upper:
                signal = -1  # Potential sell
            
            # Apply confirmation rules
            if signal != 0:
                # Only take trades during BB squeeze
                if not self.check_bb_squeeze():
                    return 0
                
                # Check volume
                if not self.check_volume_confirmation():
                    return 0
                
                # Check trend alignment
                trend_signal = self.check_trend_alignment()
                if trend_signal is not None and signal != trend_signal:
                    return 0
                
                # Check momentum
                momentum_signal = self.check_momentum()
                if momentum_signal is not None and signal != momentum_signal:
                    return 0
                
                # Check RSI confirmation
                if not self.check_rsi_confirmation(signal):
                    return 0
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0
    
    def ready_to_trade(self) -> bool:
        """Check if strategy is ready to trade"""
        return (self.df is not None and len(self.df) >= self.trend_period and
                all(indicator in self.indicators for indicator in [
                    'bb_upper_primary', 'bb_lower_primary', 'bb_middle_primary',
                    'bb_width_primary', 'rsi', 'momentum'
                ]))