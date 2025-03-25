"""
Enhanced RSI strategy with dynamic thresholds and trend confirmation.
"""

import logging
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    def __init__(self, client, symbol: str, timeframe: str, risk_manager=None,
                 rsi_period: int = 14, base_overbought: float = 70, base_oversold: float = 30,
                 trend_period: int = 200):
        super().__init__(client, symbol, timeframe, risk_manager)
        
        self.rsi_period = rsi_period
        self.base_overbought = base_overbought
        self.base_oversold = base_oversold
        self.trend_period = trend_period
        self.indicators: Dict[str, Any] = {}
        
        # Additional timeframes for confirmation
        self.timeframes = {
            'short': self.get_lower_timeframe(timeframe),
            'primary': timeframe,
            'long': self.get_higher_timeframe(timeframe)
        }
    
    def calculate_indicators(self) -> None:
        """Calculate RSI and supporting indicators"""
        if self.df is None or len(self.df) < self.trend_period:
            return
            
        try:
            # Calculate RSI for each timeframe
            for tf_name, tf_data in self.timeframes.items():
                if tf_name == 'primary':
                    data = self.df
                else:
                    data = self.client.get_historical_klines(timeframe=tf_data, limit=300)
                    if data is None or len(data) < self.trend_period:
                        continue
                
                # Calculate RSI
                close_diff = data['close'].diff()
                gains = close_diff.where(close_diff > 0, 0)
                losses = -close_diff.where(close_diff < 0, 0)
                
                avg_gains = gains.rolling(window=self.rsi_period).mean()
                avg_losses = losses.rolling(window=self.rsi_period).mean()
                
                rs = avg_gains / avg_losses
                self.indicators[f'rsi_{tf_name}'] = 100 - (100 / (1 + rs))
                
                # Calculate trend indicators
                self.indicators[f'sma_{tf_name}'] = data['close'].rolling(window=self.trend_period).mean()
                self.indicators[f'ema_{tf_name}'] = data['close'].ewm(span=self.trend_period).mean()
                
            # Calculate volatility and dynamic thresholds
            returns = self.df['close'].pct_change()
            self.indicators['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            self.indicators['atr'] = self.calculate_atr(self.df, period=14)
            
            # Calculate volume indicators
            self.indicators['volume_sma'] = self.df['volume'].rolling(window=20).mean()
            self.indicators['volume_ratio'] = self.df['volume'] / self.indicators['volume_sma']
            
            # Calculate momentum
            self.indicators['momentum'] = returns.rolling(window=10).mean() * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def get_dynamic_thresholds(self) -> tuple:
        """Calculate dynamic overbought/oversold thresholds based on volatility"""
        try:
            vol = self.indicators['volatility'].iloc[-1]
            base_range = (self.base_overbought - self.base_oversold) / 2
            
            # Adjust thresholds based on volatility
            adjustment = min(15, max(5, vol * 10))  # Cap adjustment at 15
            
            dynamic_overbought = min(85, self.base_overbought + adjustment)
            dynamic_oversold = max(15, self.base_oversold - adjustment)
            
            return dynamic_overbought, dynamic_oversold
            
        except Exception as e:
            logger.error(f"Error calculating dynamic thresholds: {e}")
            return self.base_overbought, self.base_oversold
    
    def check_volume_confirmation(self) -> bool:
        """Check if volume confirms the signal"""
        try:
            volume_ratio = self.indicators['volume_ratio'].iloc[-1]
            return volume_ratio > 1.5  # 50% above average volume
        except Exception as e:
            logger.error(f"Error checking volume: {e}")
            return False
    
    def check_trend_alignment(self) -> Optional[int]:
        """Check if price action aligns with multiple timeframe trend"""
        try:
            alignments = []
            
            for tf_name in self.timeframes:
                if (f'sma_{tf_name}' not in self.indicators or
                    f'ema_{tf_name}' not in self.indicators):
                    continue
                
                current_price = self.df['close'].iloc[-1]
                sma = self.indicators[f'sma_{tf_name}'].iloc[-1]
                ema = self.indicators[f'ema_{tf_name}'].iloc[-1]
                rsi = self.indicators[f'rsi_{tf_name}'].iloc[-1]
                
                # Determine trend direction
                if current_price > sma and current_price > ema and rsi > 50:
                    alignments.append(1)  # Uptrend
                elif current_price < sma and current_price < ema and rsi < 50:
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
        Generate trading signal based on enhanced RSI strategy.
        Returns:
            int: 1 for buy, -1 for sell, 0 for no action
        """
        if not self.ready_to_trade():
            return 0
            
        try:
            # Get current RSI and dynamic thresholds
            current_rsi = self.indicators['rsi_primary'].iloc[-1]
            overbought, oversold = self.get_dynamic_thresholds()
            
            signal = 0
            
            # Generate initial signal based on RSI
            if current_rsi < oversold:
                signal = 1  # Potential buy
            elif current_rsi > overbought:
                signal = -1  # Potential sell
            
            # Apply confirmation rules
            if signal != 0:
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
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0
    
    def ready_to_trade(self) -> bool:
        """Check if strategy is ready to trade"""
        return (self.df is not None and len(self.df) >= self.trend_period and
                all(indicator in self.indicators for indicator in 
                    ['rsi_primary', 'volatility', 'momentum']))