"""
Binance API client for interacting with Binance Futures.
"""

import logging
import math
import time
import pandas as pd
from functools import wraps
from binance.client import Client
from binance.enums import *
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def rate_limited(max_per_second: float):
    """Rate limiting decorator"""
    min_interval = 1.0 / float(max_per_second)
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            result = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return result
        return wrapper
    return decorator

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2
            return None
        return wrapper
    return decorator

class BinanceClient:
    """Binance API client wrapper"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """Initialize Binance client"""
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.symbol = "BTCUSDT"  # Fixed to BTCUSDT
        self.order_precision = {}  # Cache for symbol precision
        self._init_symbol_precision()
        
    def _init_symbol_precision(self):
        """Initialize symbol precision for order quantities"""
        try:
            info = self.client.get_symbol_info(self.symbol)
            if info:
                filters = {f['filterType']: f for f in info['filters']}
                if 'LOT_SIZE' in filters:
                    step_size = float(filters['LOT_SIZE']['stepSize'])
                    self.order_precision[self.symbol] = int(round(-math.log10(step_size)))
        except Exception as e:
            logger.error(f"Error initializing symbol precision: {e}")
            self.order_precision[self.symbol] = 5  # Default precision
    
    @retry_on_error()
    @rate_limited(10.0)
    def get_ticker_price(self, symbol=None):
        """Get current price with error handling"""
        try:
            symbol = symbol or self.symbol
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return None
    
    @retry_on_error(max_retries=3)
    @rate_limited(10.0)
    def get_historical_klines(self, symbol=None, timeframe="1h", limit=100, start_time=None, end_time=None):
        """Get historical candlestick data with error handling and rate limiting"""
        try:
            symbol = symbol or self.symbol
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    @retry_on_error()
    def get_account_balance(self):
        """Get account balance with error handling"""
        try:
            account = self.client.futures_account_balance()
            for balance in account:
                if balance['asset'] == 'USDT':
                    return float(balance['balance'])
            return None
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return None
    
    @retry_on_error()
    def get_open_positions(self):
        """Get open positions with error handling"""
        try:
            positions = self.client.futures_position_information()
            return [pos for pos in positions if float(pos['positionAmt']) != 0]
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []
    
    @retry_on_error()
    def create_market_order(self, side, quantity):
        """Create market order with proper quantity rounding"""
        try:
            precision = self.order_precision.get(self.symbol, 5)
            quantity = round(quantity, precision)
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            return order
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None
    
    @retry_on_error()
    def create_stop_loss_order(self, side, quantity, stop_price):
        """Create stop loss order with proper price and quantity rounding"""
        try:
            precision = self.order_precision.get(self.symbol, 5)
            quantity = round(quantity, precision)
            stop_price = round(stop_price, 2)  # Price precision usually 2 for USDT pairs
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='STOP_MARKET',
                stopPrice=stop_price,
                quantity=quantity,
                reduceOnly=True
            )
            return order
        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            return None
    
    @retry_on_error()
    def create_take_profit_order(self, side, quantity, take_profit_price):
        """Create take profit order with proper price and quantity rounding"""
        try:
            precision = self.order_precision.get(self.symbol, 5)
            quantity = round(quantity, precision)
            take_profit_price = round(take_profit_price, 2)
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit_price,
                quantity=quantity,
                reduceOnly=True
            )
            return order
        except Exception as e:
            logger.error(f"Error creating take profit order: {e}")
            return None
    
    @retry_on_error()
    def cancel_order(self, order_id):
        """Cancel order with error handling"""
        try:
            return self.client.futures_cancel_order(
                symbol=self.symbol,
                orderId=order_id
            )
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return None
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol"""
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling all orders for {symbol}: {e}")
            return False
    
    def close_all_positions(self, symbol: str) -> bool:
        """Close all positions for a symbol"""
        try:
            positions = self.get_open_positions()
            for position in positions:
                if position['symbol'] == symbol and float(position['positionAmt']) != 0:
                    side = "SELL" if float(position['positionAmt']) > 0 else "BUY"
                    self.create_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(float(position['positionAmt'])),
                        reduce_only=True
                    )
            return True
        except Exception as e:
            logger.error(f"Error closing positions for {symbol}: {e}")
            return False