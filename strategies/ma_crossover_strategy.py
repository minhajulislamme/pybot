"""
Enhanced Moving Average Crossover strategy with volume and trend strength analysis.
"""

import logging
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MACrossoverStrategy(BaseStrategy):
    def __init__(self, client, symbol: str, timeframe: str, risk_manager=None,
                 fast_ma_period: int = 20, slow_ma_period: int = 50, ma_type: str = 'ema',
                 volume_factor: float = 2.0):
        super().__init__(client, symbol, timeframe, risk_manager)
        
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.ma_type = ma_type.lower()
        self.volume_factor = volume_factor
        
        # Additional timeframes for confirmation
        self.timeframes = {
            'short': self.get_lower_timeframe(timeframe),
            'primary': timeframe,
            'long': self.get_higher_timeframe(timeframe)
        }
        
        # Store indicators
        self.indicators: Dict[str, Any] = {}
        
    def calculate_indicators(self) -> None:
        """Calculate all required indicators"""
        if self.df is None or len(self.df) < self.slow_ma_period:
            return
            
        try:
            # Calculate MAs for primary timeframe
            for tf_name, tf_data in self.timeframes.items():
                if tf_name == 'primary':
                    data = self.df
                else:
                    data = self.client.get_historical_klines(timeframe=tf_data, limit=100)
                    if data is None or len(data) < self.slow_ma_period:
                        continue
                
                # Calculate fast MA
                self.indicators[f'fast_ma_{tf_name}'] = (
                    data['close'].ewm(span=self.fast_ma_period).mean() if self.ma_type == 'ema'
                    else data['close'].rolling(window=self.fast_ma_period).mean()
                )
                
                # Calculate slow MA
                self.indicators[f'slow_ma_{tf_name}'] = (
                    data['close'].ewm(span=self.slow_ma_period).mean() if self.ma_type == 'ema'
                    else data['close'].rolling(window=self.slow_ma_period).mean()
                )
                
                # Calculate trend strength
                self.indicators[f'trend_strength_{tf_name}'] = (
                    self.indicators[f'fast_ma_{tf_name}'] - self.indicators[f'slow_ma_{tf_name}']
                ) / self.indicators[f'slow_ma_{tf_name}'] * 100
            
            # Calculate volume indicators
            self.indicators['avg_volume'] = self.df['volume'].rolling(window=20).mean()
            self.indicators['volume_std'] = self.df['volume'].rolling(window=20).std()
            
            # Calculate momentum and volatility
            returns = self.df['close'].pct_change()
            self.indicators['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            self.indicators['momentum'] = returns.rolling(window=10).mean() * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    def check_volume_confirmation(self) -> bool:
        """Check if volume confirms the signal"""
        try:
            current_volume = self.df['volume'].iloc[-1]
            avg_volume = self.indicators['avg_volume'].iloc[-1]
            vol_std = self.indicators['volume_std'].iloc[-1]
            
            return (current_volume > avg_volume * self.volume_factor and
                   current_volume > avg_volume + (2 * vol_std))
        except Exception as e:
            logger.error(f"Error checking volume: {e}")
            return False
    
    def check_trend_alignment(self) -> Optional[int]:
        """Check if multiple timeframes show trend alignment"""
        try:
            alignments = []
            
            for tf_name in self.timeframes:
                if (f'trend_strength_{tf_name}' not in self.indicators or
                    f'fast_ma_{tf_name}' not in self.indicators or
                    f'slow_ma_{tf_name}' not in self.indicators):
                    continue
                
                trend_strength = self.indicators[f'trend_strength_{tf_name}'].iloc[-1]
                fast_ma = self.indicators[f'fast_ma_{tf_name}'].iloc[-1]
                slow_ma = self.indicators[f'slow_ma_{tf_name}'].iloc[-1]
                
                if fast_ma > slow_ma and trend_strength > 0:
                    alignments.append(1)  # Uptrend
                elif fast_ma < slow_ma and trend_strength < 0:
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
        """Check momentum direction"""
        try:
            momentum = self.indicators['momentum'].iloc[-1]
            volatility = self.indicators['volatility'].iloc[-1]
            
            # Adjust momentum threshold based on volatility
            threshold = 0.1 * (1 + volatility)  # Dynamic threshold
            
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
        Generate trading signal based on enhanced MA crossover strategy.
        Returns:
            int: 1 for buy, -1 for sell, 0 for no action
        """
        if not self.ready_to_trade():
            return 0
            
        try:
            # Get current indicators for primary timeframe
            fast_ma = self.indicators['fast_ma_primary'].iloc[-1]
            slow_ma = self.indicators['slow_ma_primary'].iloc[-1]
            trend_strength = self.indicators['trend_strength_primary'].iloc[-1]
            
            signal = 0
            
            # Generate initial signal based on MA crossover
            if fast_ma > slow_ma and trend_strength > 0:
                signal = 1  # Potential buy
            elif fast_ma < slow_ma and trend_strength < 0:
                signal = -1  # Potential sell
                
            # Apply confirmation rules
            if signal != 0:
                # Check volume
                if not self.check_volume_confirmation():
                    return 0
                
                # Check multiple timeframe alignment
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
        return (self.df is not None and len(self.df) >= self.slow_ma_period and
                all(indicator in self.indicators for indicator in 
                    ['fast_ma_primary', 'slow_ma_primary', 'trend_strength_primary']))