"""
Configuration module for trading bot.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class BotConfig:
    """
    Configuration class for the trading bot.
    Loads settings from environment variables and provides defaults.
    """
    
    def __init__(self, test_mode=True):
        """
        Initialize configuration.
        
        Args:
            test_mode (bool): If True, use testnet API credentials
        """
        self.test_mode = test_mode
        self.testnet = test_mode  # Add testnet property for compatibility
        
        # API credentials
        if test_mode:
            self.api_key = os.getenv('BINANCE_TESTNET_API_KEY', '')
            self.api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '')
            self.testnet_api_key = self.api_key
            self.testnet_api_secret = self.api_secret
            self.ws_url = "wss://fstream.binancefuture.com"  # Updated testnet WebSocket URL
        else:
            self.api_key = os.getenv('BINANCE_API_KEY', '')
            self.api_secret = os.getenv('BINANCE_API_SECRET', '')
            self.ws_url = "wss://fstream.binance.com"  # Production WebSocket URL
            
        # Trading symbols - only BTCUSDT
        self.symbols = ["BTCUSDT"]
        
        # Trading timeframes - only 1-hour timeframe
        self.timeframes = ["1h"]
        
        # Default strategy 
        self.default_strategy = os.getenv('DEFAULT_STRATEGY', 'MAcrossover')
        
        # Order type
        self.order_type = os.getenv('ORDER_TYPE', 'MARKET')
        
        # Risk management settings
        self.max_risk_per_trade = 1.0
        self.default_stop_loss = 2.0
        self.default_take_profit = 4.0
        self.use_trailing_stop = True
        self.trailing_activation_pct = float(os.getenv('TRAILING_STOP_ACTIVATION_PCT', '1.0'))
        self.max_open_positions = 1
        
        # Telegram settings
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'True').lower() == 'true'
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """
        Validate critical configuration settings.
        """
        if not self.api_key or not self.api_secret:
            logger.warning(
                f"API credentials missing for {'testnet' if self.test_mode else 'live'} mode. "
                "Please check your environment variables."
            )
            
        if not self.symbols:
            logger.warning("No trading symbols specified, using default: BTCUSDT")
            self.symbols = ['BTCUSDT']
            
        if not self.timeframes:
            logger.warning("No trading timeframes specified, using default: 1h")
            self.timeframes = ['1h']
            
        if self.telegram_enabled and (not self.telegram_token or not self.telegram_chat_id):
            logger.warning("Telegram notifications enabled but token or chat ID missing")
            self.telegram_enabled = False
            
    def get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy from environment variables.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            dict: Strategy parameters
        """
        params = {}
        
        if strategy_name == "MAcrossover":
            params['fast_ma_period'] = 9
            params['slow_ma_period'] = 21
            params['ma_type'] = os.getenv('MA_TYPE', 'sma')
            
        elif strategy_name == "RSI":
            params['rsi_period'] = 14
            params['overbought'] = 70
            params['oversold'] = 30
            
        elif strategy_name == "BollingerBands":
            params['bb_period'] = 20
            params['bb_std_dev'] = 2.0
            params['use_squeeze'] = os.getenv('BB_USE_SQUEEZE', 'True').lower() == 'true'
        
        return params
    

# Create a default instance
config = BotConfig()