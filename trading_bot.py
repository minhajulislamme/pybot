import logging
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import signal
import sys
import concurrent.futures
import schedule

from binance_client import BinanceClient
from websocket_client import BinanceWebsocket
from strategy import create_strategy
from risk_manager import RiskManager
from telegram_notifier import TelegramNotifier
from trade_validator import TradeValidator
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Trading bot class for a single cryptocurrency pair"""
    
    def __init__(self, trading_config):
        self.client = BinanceClient()
        self.telegram = TelegramNotifier()
        
        # Initialize trading parameters from config
        self.symbol = trading_config.get("symbol")
        self.timeframe = config.BACKTEST_TIMEFRAME
        self.leverage = trading_config.get("leverage", config.DEFAULT_LEVERAGE)
        self.strategy_name = trading_config.get("strategy", config.DEFAULT_STRATEGY)
        self.risk_percentage = trading_config.get("risk_percentage", config.DEFAULT_RISK_PERCENTAGE)
        self.stop_loss_percentage = trading_config.get("stop_loss_percentage", config.DEFAULT_STOP_LOSS_PERCENTAGE)
        self.take_profit_percentage = trading_config.get("take_profit_percentage", config.DEFAULT_TAKE_PROFIT_PERCENTAGE)
        
        # Initialize risk manager with symbol-specific parameters
        self.risk_manager = RiskManager(
            self.client, 
            risk_percentage=self.risk_percentage,
            stop_loss_percentage=self.stop_loss_percentage,
            take_profit_percentage=self.take_profit_percentage
        )
        
        # Create strategy instance with symbol-specific parameters
        strategy_params = {}
        
        if self.strategy_name == "SMA_CROSSOVER":
            strategy_params = {
                'short_period': trading_config.get("short_sma", config.DEFAULT_SHORT_SMA),
                'long_period': trading_config.get("long_sma", config.DEFAULT_LONG_SMA)
            }
        elif self.strategy_name == "RSI":
            strategy_params = {
                'period': trading_config.get("rsi_period", config.DEFAULT_RSI_PERIOD),
                'overbought': trading_config.get("rsi_overbought", config.DEFAULT_RSI_OVERBOUGHT),
                'oversold': trading_config.get("rsi_oversold", config.DEFAULT_RSI_OVERSOLD)
            }
        elif self.strategy_name == "MACD":
            strategy_params = {
                'fast_period': trading_config.get("fast_period", 12),
                'slow_period': trading_config.get("slow_period", 26),
                'signal_period': trading_config.get("signal_period", 9)
            }
            
        self.strategy = create_strategy(self.strategy_name, self.symbol, self.timeframe, **strategy_params)
        
        # Initialize trading state
        self.is_running = False
        self.initial_balance = 0
        self.current_position = None
        self.current_orders = {}
        self.klines_data = []
        self.klines_df = None
        self.latest_bid = 0
        self.latest_ask = 0
        self.latest_mark_price = 0
        
        # Trading performance stats
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'start_balance': 0,
            'current_balance': 0,
            'max_drawdown': 0
        }
        
        # Daily stats reset
        self.daily_stats = self.trading_stats.copy()
        self.daily_reset_time = datetime.now()
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Trading bot initialized for {self.symbol} using {self.strategy_name} strategy")
    
    def initialize_websocket(self):
        """Initialize and connect to Binance websocket"""
        try:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"[{self.symbol}] Initializing WebSocket connection (attempt {retry_count+1}/{max_retries})")
                    
                    # Create new websocket client
                    self.ws = BinanceWebsocket(self.symbol, self.on_websocket_message, self.on_websocket_close)
                    
                    # Connect to WebSocket
                    self.ws.connect()
                    
                    # Because we added connection confirmation in the BinanceWebsocket class,
                    # if we reach this point, then the connection was successful
                    
                    # Subscribe to essential data streams only
                    logger.info(f"[{self.symbol}] WebSocket connected, subscribing to data streams")
                    time.sleep(1)  # Short delay before subscribing
                    
                    self.ws.subscribe_kline(self.symbol, '1m')
                    time.sleep(1)
                    self.ws.subscribe_book_ticker(self.symbol)
                    
                    logger.info(f"[{self.symbol}] WebSocket connection initialized with essential data streams")
                    
                    # Start a websocket health check timer
                    self.start_websocket_health_check()
                    
                    return True  # Success
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"[{self.symbol}] WebSocket connection attempt {retry_count} failed: {e}")
                    if retry_count < max_retries:
                        time.sleep(5)  # Wait before retrying
                    else:
                        logger.error(f"[{self.symbol}] Failed to connect to WebSocket after {max_retries} attempts")
                        return False
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error initializing websocket: {e}"
            logger.error(error_msg)
            
            # Don't send error notification for WebSocket issues as we'll fall back to REST API
            return False
    
    def start_websocket_health_check(self):
        """Start a timer to regularly check WebSocket health"""
        def check_health():
            if not self.is_running:
                return
                
            if hasattr(self, 'ws') and self.ws:
                if not self.ws.is_alive():
                    logger.warning(f"[{self.symbol}] WebSocket connection lost, attempting to reconnect...")
                    try:
                        self.initialize_websocket()
                    except Exception as e:
                        logger.error(f"[{self.symbol}] Failed to reconnect websocket: {e}")
            
            # Schedule next health check in 60 seconds
            if self.is_running:
                threading.Timer(60, check_health).start()
                
        # Start first health check after a delay
        threading.Timer(60, check_health).start()
    
    def on_websocket_message(self, message):
        """Process incoming websocket message"""
        try:
            # Check if it's a kline/candlestick message
            if 'k' in message:
                self.process_kline_message(message)
                
                # Log a more readable summary of the kline data
                kline = message['k']
                symbol = kline.get('s', 'unknown')
                interval = kline.get('i', 'unknown')
                close_time = datetime.fromtimestamp(kline.get('T', 0)/1000).strftime('%Y-%m-%d %H:%M:%S')
                is_closed = kline.get('x', False)
                
                logger.info(f"[{self.symbol}] Received kline: {symbol} {interval} Close time: {close_time} Closed: {is_closed}")
                logger.info(f"[{self.symbol}] Kline data: Open: {kline.get('o')}, High: {kline.get('h')}, Low: {kline.get('l')}, Close: {kline.get('c')}")
            
            # Check if it's a book ticker message (best bid/ask)
            elif 'b' in message and 'a' in message and 's' in message:
                if isinstance(message['b'], str) and isinstance(message['a'], str):
                    self.process_book_ticker_message(message)
                    
                    # Log book ticker data
                    symbol = message.get('s', 'unknown')
                    bid = message.get('b', 'N/A')
                    ask = message.get('a', 'N/A')
                    bid_qty = message.get('B', 'N/A')
                    ask_qty = message.get('A', 'N/A')
                    
                    logger.info(f"[{self.symbol}] Book Ticker: {symbol} Bid: {bid} ({bid_qty}), Ask: {ask} ({ask_qty})")
                
            # Check if it's a mark price message
            elif 'e' in message and message['e'] == 'markPriceUpdate' and 'p' in message:
                if isinstance(message['p'], str):
                    self.process_mark_price_message(message)
                    
                    # Log mark price data
                    symbol = message.get('s', 'unknown')
                    mark_price = message.get('p', 'N/A')
                    funding_rate = message.get('r', 'N/A')
                    next_funding_time = message.get('T', 0)
                    
                    if next_funding_time > 0:
                        funding_time = datetime.fromtimestamp(next_funding_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        funding_time = 'N/A'
                    
                    logger.info(f"[{self.symbol}] Mark Price: {symbol} Price: {mark_price}, Funding Rate: {funding_rate}, Next Funding: {funding_time}")
            
            # Check if it's a trade message
            elif 'e' in message and message['e'] == 'trade':
                # Log trade data
                symbol = message.get('s', 'unknown')
                price = message.get('p', 'N/A')
                quantity = message.get('q', 'N/A')
                buyer_order_id = message.get('b', 'N/A')
                seller_order_id = message.get('a', 'N/A')
                trade_time = datetime.fromtimestamp(message.get('T', 0)/1000).strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"[{self.symbol}] Trade: {symbol} Price: {price}, Quantity: {quantity}, Time: {trade_time}")
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing websocket message: {e}")
            # Log the problematic message
            try:
                logger.error(f"[{self.symbol}] Message causing error: {json.dumps(message)[:200]}...")
            except:
                logger.error(f"[{self.symbol}] Could not log message content")
    
    def on_websocket_close(self):
        """Handle websocket connection close"""
        logger.warning(f"[{self.symbol}] WebSocket connection closed, attempting to reconnect...")
        self.telegram.send_message(f"🔄 [{self.symbol}] WebSocket connection lost, reconnecting...")
        time.sleep(5)
        try:
            self.initialize_websocket()
        except Exception as e:
            error_msg = f"[{self.symbol}] Failed to reconnect to websocket: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
    
    def process_kline_message(self, message):
        """Process kline/candlestick data from websocket"""
        try:
            kline = message['k']
            
            # Only process if the candle is closed
            if kline['x']:
                symbol = kline.get('s', '').upper()
                
                # Only process if symbol matches
                if symbol != self.symbol:
                    return
                    
                logger.info(f"[{self.symbol}] Closed candle: {kline['t']} - Open: {kline['o']}, Close: {kline['c']}")
                
                # Add to klines data
                with self.lock:
                    self.add_kline_to_data(kline)
                    self.update_strategy()
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing kline message: {e}")
        
    def add_kline_to_data(self, kline):
        """Add a new kline to the internal data storage"""
        # Format the kline to match our expected format
        formatted_kline = [
            kline['t'],         # Open time
            kline['o'],         # Open
            kline['h'],         # High
            kline['l'],         # Low
            kline['c'],         # Close
            kline['v'],         # Volume
            kline['T'],         # Close time
            kline['q'],         # Quote asset volume
            kline['n'],         # Number of trades
            kline['V'],         # Taker buy base asset volume
            kline['Q'],         # Taker buy quote asset volume
            '0'                 # Ignore
        ]
        
        # Add to klines data
        self.klines_data.append(formatted_kline)
        
        # Keep only the most recent 1000 candles
        if len(self.klines_data) > 1000:
            self.klines_data.pop(0)
        
        # Convert to DataFrame
        self.klines_df = self.strategy.prepare_data(self.klines_data)
    
    def process_book_ticker_message(self, message):
        """Process book ticker data (best bid/ask)"""
        try:
            # Store the latest bid/ask prices
            symbol = message.get('s', '').upper()
            
            # Only process if symbol matches
            if symbol != self.symbol:
                return
                
            self.latest_bid = float(message['b'])
            self.latest_ask = float(message['a'])
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing book ticker: {e}")
    
    def process_mark_price_message(self, message):
        """Process mark price updates"""
        try:
            # Store the latest mark price
            symbol = message.get('s', '').upper()
            
            # Only process if symbol matches
            if symbol != self.symbol:
                return
                
            self.latest_mark_price = float(message['p'])
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing mark price: {e}")
    
    def process_trade_message(self, message):
        """Process real-time trade data"""
        try:
            if 'p' in message and isinstance(message['p'], str) and 'q' in message and isinstance(message['q'], str):
                price = float(message['p'])
                quantity = float(message['q'])
                
                # Update the latest price in memory
                self.latest_price = price
                
                # Add trade to a counter for volume analysis
                if not hasattr(self, 'trade_volume'):
                    self.trade_volume = 0
                    self.trade_count = 0
                    self.last_volume_reset = time.time()
                    
                self.trade_volume += quantity
                self.trade_count += 1
                
                # Reset volume counters every minute
                if time.time() - self.last_volume_reset > 60:
                    logger.info(f"[{self.symbol}] 1-minute volume: {self.trade_volume} | Trades: {self.trade_count}")
                    self.trade_volume = 0
                    self.trade_count = 0
                    self.last_volume_reset = time.time()
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing trade message: {e}")
    
    def process_depth_message(self, message):
        """Process order book depth updates"""
        try:
            # Initialize order book if it doesn't exist
            if not hasattr(self, 'order_book'):
                self.order_book = {'bids': {}, 'asks': {}}
                
            # Update bids in the order book - safely handle lists
            if 'b' in message and isinstance(message['b'], list):
                for bid in message['b']:
                    if len(bid) >= 2 and all(isinstance(x, str) for x in bid[:2]):
                        price = float(bid[0])
                        quantity = float(bid[1])
                        
                        if quantity == 0:
                            # Remove the price level
                            if price in self.order_book['bids']:
                                del self.order_book['bids'][price]
                        else:
                            # Update the price level
                            self.order_book['bids'][price] = quantity
                    
            # Update asks in the order book - safely handle lists
            if 'a' in message and isinstance(message['a'], list):
                for ask in message['a']:
                    if len(ask) >= 2 and all(isinstance(x, str) for x in ask[:2]):
                        price = float(ask[0])
                        quantity = float(ask[1])
                        
                        if quantity == 0:
                            # Remove the price level
                            if price in self.order_book['asks']:
                                del self.order_book['asks'][price]
                        else:
                            # Update the price level
                            self.order_book['asks'][price] = quantity
                    
            # Calculate order book imbalance only if we have data
            if self.order_book['bids'] and self.order_book['asks']:
                bid_total = sum(self.order_book['bids'].values())
                ask_total = sum(self.order_book['asks'].values())
                
                if bid_total + ask_total > 0:
                    self.book_imbalance = (bid_total - ask_total) / (bid_total + ask_total)
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing depth message: {e}")

    def update_strategy(self):
        """Update strategy with latest data and process signals"""
        if self.klines_df is None or len(self.klines_df) < 50:  # Using a default minimum value
            logger.info(f"[{self.symbol}] Not enough data for strategy calculations yet")
            return
        
        # Calculate signals
        signals_df = self.strategy.calculate_signals(self.klines_df)
        
        # Check the latest signal
        latest_signal = signals_df.iloc[-1]['signal']
        
        if (latest_signal != 0):
            # Process the trading signal
            self.process_trading_signal(latest_signal, signals_df)
    
    def process_trading_signal(self, signal, data):
        """Process a trading signal and execute trades accordingly"""
        try:
            # Get current position info
            position = self.client.get_position_info(self.symbol)
            current_position_size = float(position['positionAmt']) if position else 0
            
            # Current price from real-time websocket data
            current_price = self.get_real_time_price()
            if current_price <= 0:
                logger.error(f"[{self.symbol}] Invalid price received, cannot execute trade")
                return
                
            # Initialize trade validator if not exists
            if not hasattr(self, 'trade_validator'):
                self.trade_validator = TradeValidator(self.client, self.risk_manager)
            
            # Position side
            position_side = "BUY" if signal > 0 else "SELL"
            
            # Get additional confirmation from multiple timeframes
            trend_confirmation = self.get_trend_confirmation()
            
            # Add order book imbalance to the confirmation
            if hasattr(self, 'book_imbalance'):
                # Positive imbalance means more bids than asks (bullish)
                # Negative imbalance means more asks than bids (bearish)
                if self.book_imbalance > 0.1 and signal > 0:
                    trend_confirmation *= 1.2  # Boost bullish confirmation
                elif self.book_imbalance < -0.1 and signal < 0:
                    trend_confirmation *= 1.2  # Boost bearish confirmation
            
            # Only trade when signal aligns with the trend confirmation
            signal_strength = signal * trend_confirmation
            
            if signal_strength <= 0:
                logger.info(f"[{self.symbol}] Signal {signal} rejected due to trend confirmation {trend_confirmation}")
                return
                
            # Check market conditions
            market_condition = self.analyze_market_condition()
            
            if market_condition == "VOLATILE":
                logger.info(f"[{self.symbol}] Market condition volatile, reducing position size")
                # Reduce position size in volatile markets
                risk_multiplier = 0.5
            elif market_condition == "TRENDING":
                logger.info(f"[{self.symbol}] Market condition trending, increasing position size")
                # Increase position size in trending markets
                risk_multiplier = 1.2
            else:
                risk_multiplier = 1.0
            
            if signal == 1:  # Buy signal
                if current_position_size <= 0:  # No position or short position
                    # Close any existing short position
                    if current_position_size < 0:
                        self.close_position(position, current_price)
                    
                    # Calculate volatility-adjusted position size
                    base_risk = self.risk_percentage * risk_multiplier
                    quantity = self.calculate_volatility_adjusted_position(self.symbol, current_price, base_risk)
                    
                    # Fall back to standard calculation if volatility calc fails
                    if not quantity:
                        quantity = self.trade_validator.validate_and_calculate_position(self.symbol, current_price)
                    
                    if not quantity:
                        logger.warning(f"[{self.symbol}] Invalid position size calculated")
                        return
                    
                    # Open long position
                    order = self.client.place_market_order(self.symbol, "BUY", quantity)
                    
                    if order:
                        # Calculate and set stop loss and take profit levels
                        sl_price = self.risk_manager.calculate_stop_loss_price(current_price, "BUY")
                        tp_price = self.risk_manager.calculate_take_profit_price(current_price, "BUY")
                        
                        # Check if trend is strong enough for trailing stop
                        if trend_confirmation >= 0.5:
                            # Set trailing stop instead of fixed stop loss
                            activation_price = current_price * 1.005  # Activate when price moves 0.5% in favor
                            trailing_stop = self.set_trailing_stop_loss(self.symbol, "BUY", quantity, activation_price)
                            
                            # If trailing stop fails, fall back to regular stop loss
                            if not trailing_stop:
                                sl_order = self.client.place_stop_loss(self.symbol, "BUY", quantity, sl_price)
                        else:
                            # Use regular stop loss
                            sl_order = self.client.place_stop_loss(self.symbol, "BUY", quantity, sl_price)
                        
                        # Place take profit order
                        tp_order = self.client.place_take_profit(self.symbol, "BUY", quantity, tp_price)
                        
                        # Send notifications
                        self.telegram.send_trade_notification(self.symbol, "BUY", quantity, current_price)
                        self.telegram.send_sl_tp_notification(self.symbol, "BUY", sl_price, tp_price)
                
            elif signal == -1:  # Sell signal
                if current_position_size >= 0:  # No position or long position
                    # Close any existing long position
                    if current_position_size > 0:
                        self.close_position(position, current_price)
                    
                    # Calculate volatility-adjusted position size
                    base_risk = self.risk_percentage * risk_multiplier
                    quantity = self.calculate_volatility_adjusted_position(self.symbol, current_price, base_risk)
                    
                    # Fall back to standard calculation if volatility calc fails
                    if not quantity:
                        quantity = self.trade_validator.validate_and_calculate_position(self.symbol, current_price)
                    
                    if not quantity:
                        logger.warning(f"[{self.symbol}] Invalid position size calculated")
                        return
                    
                    # Open short position
                    order = self.client.place_market_order(self.symbol, "SELL", quantity)
                    
                    if order:
                        # Calculate and set stop loss and take profit levels
                        sl_price = self.risk_manager.calculate_stop_loss_price(current_price, "SELL")
                        tp_price = self.risk_manager.calculate_take_profit_price(current_price, "SELL")
                        
                        # Check if trend is strong enough for trailing stop
                        if trend_confirmation <= -0.5:
                            # Set trailing stop instead of fixed stop loss
                            activation_price = current_price * 0.995  # Activate when price moves 0.5% in favor
                            trailing_stop = self.set_trailing_stop_loss(self.symbol, "SELL", quantity, activation_price)
                            
                            # If trailing stop fails, fall back to regular stop loss
                            if not trailing_stop:
                                sl_order = self.client.place_stop_loss(self.symbol, "SELL", quantity, sl_price)
                        else:
                            # Use regular stop loss
                            sl_order = self.client.place_stop_loss(self.symbol, "SELL", quantity, sl_price)
                        
                        # Place take profit order
                        tp_order = self.client.place_take_profit(self.symbol, "SELL", quantity, tp_price)
                        
                        # Send notifications
                        self.telegram.send_trade_notification(self.symbol, "SELL", quantity, current_price)
                        self.telegram.send_sl_tp_notification(self.symbol, "SELL", sl_price, tp_price)
            
            # Update trading stats
            self.update_trading_stats()
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error executing trade: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
    
    def update_trading_stats(self):
        """Update trading performance statistics"""
        try:
            # Get account balance
            balance = self.client.get_account_balance()
            
            # Initialize start balance if not already set
            if self.trading_stats['start_balance'] == 0:
                self.trading_stats['start_balance'] = balance
                self.daily_stats['start_balance'] = balance
            
            # Update current balance
            self.trading_stats['current_balance'] = balance
            self.daily_stats['current_balance'] = balance
            
            # Calculate profit/loss
            pnl = balance - self.trading_stats['start_balance']
            self.trading_stats['total_profit_loss'] = pnl
            
            daily_pnl = balance - self.daily_stats['start_balance']
            self.daily_stats['total_profit_loss'] = daily_pnl
            
            # Calculate percentage PnL
            pnl_percentage = (pnl / self.trading_stats['start_balance']) * 100 if self.trading_stats['start_balance'] > 0 else 0
            
            # Send balance update every 10 trades or on significant changes
            if (self.trading_stats['total_trades'] % 10 == 0 or 
                abs(pnl_percentage) > 5 or  # Significant PnL change
                abs(pnl) > 100):  # Significant absolute PnL change
                self.telegram.send_balance_update(balance, pnl, pnl_percentage)
            
            # Calculate and update max drawdown
            if pnl < 0 and abs(pnl) > self.trading_stats['max_drawdown']:
                self.trading_stats['max_drawdown'] = abs(pnl)
                
            if daily_pnl < 0 and abs(daily_pnl) > self.daily_stats['max_drawdown']:
                self.daily_stats['max_drawdown'] = abs(daily_pnl)
            
            logger.info(f"[{self.symbol}] Updated trading stats: {self.trading_stats}")
            
            # Check if we need to reset daily stats (a new day has begun)
            current_time = datetime.now()
            if current_time.date() > self.daily_reset_time.date():
                # Send daily summary before reset
                self.telegram.send_daily_summary(self.daily_stats)
                
                # Reset daily stats
                self.daily_stats = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit_loss': 0,
                    'start_balance': balance,
                    'current_balance': balance,
                    'max_drawdown': 0
                }
                
                # Update reset time to today
                self.daily_reset_time = current_time
                logger.info(f"[{self.symbol}] Daily trading stats reset")
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error updating trading stats: {e}")
    
    def fetch_initial_data(self):
        """Fetch initial historical data for strategy calculation"""
        try:
            # Calculate how many candles we need based on strategy parameters (using a default here)
            required_candles = 200  # A reasonable default
            
            # Map timeframe to interval string expected by Binance API
            interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            interval = interval_map.get(self.timeframe, '1h')
            
            # Use REST API to fetch historical klines
            try:
                # Fetch historical klines
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = end_time - (required_candles * 60 * 60 * 1000)  # Approximate based on hourly candles
                
                klines = self.client.get_historical_klines(
                    self.symbol,
                    interval,
                    start_time,
                    end_time
                )
                
                logger.info(f"[{self.symbol}] Fetched {len(klines)} initial historical candles")
                
                # Store klines data
                self.klines_data = klines
                
                # Convert to DataFrame
                self.klines_df = self.strategy.prepare_data(klines)
            except Exception as e:
                logger.error(f"[{self.symbol}] Error fetching historical data: {e}")
                # Create empty DataFrame as fallback
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                          'close_time', 'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                self.klines_df = pd.DataFrame(columns=columns)
                self.klines_data = []
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error fetching initial data: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
    
    def start(self):
        """Start the trading bot for a specific symbol"""
        try:
            logger.info(f"[{self.symbol}] Starting trading bot...")
            
            # Fetch initial historical data
            self.fetch_initial_data()
            
            try:
                # Initialize websocket connection
                self.initialize_websocket()
            except Exception as e:
                logger.error(f"[{self.symbol}] WebSocket initialization failed, continuing with REST API: {e}")
            
            # Record initial account balance
            balance = self.client.get_account_balance()
            self.trading_stats['start_balance'] = balance
            self.trading_stats['current_balance'] = balance
            logger.info(f"[{self.symbol}] Initial account balance: {balance} USDT")
            
            # Set leverage for this symbol
            self.client.set_leverage(self.symbol, self.leverage)
            
            # Notify about bot start with account details
            self.telegram.send_message(f"🚀 <b>[{self.symbol}] Bot Started</b>\n\nStrategy: {self.strategy_name}\nLeverage: {self.leverage}x\nRisk: {self.risk_percentage}%")
            
            # Mark the bot as running
            self.is_running = True
            
            # Main trading loop
            while self.is_running:
                # Check if websocket is still connected
                if not self.ws.is_alive():
                    logger.warning(f"[{self.symbol}] WebSocket connection lost, reconnecting...")
                    self.initialize_websocket()
                
                # Check if there's new data via REST API periodically if WebSocket is not working
                if not hasattr(self, 'ws') or not self.ws or not self.ws.is_alive():
                    # Periodically fetch latest candle via REST API as a fallback
                    now = datetime.now()
                    if not hasattr(self, 'last_rest_update') or (now - self.last_rest_update).total_seconds() > 60:
                        try:
                            self.update_latest_data_via_rest()
                            self.last_rest_update = now
                        except Exception as e:
                            logger.error(f"[{self.symbol}] Error updating data via REST: {e}")
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
        except Exception as e:
            error_msg = f"[{self.symbol}] Error in trading bot: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
            self.stop()
    
    def stop(self):
        """Stop the trading bot and clean up resources"""
        if not self.is_running:
            return
            
        logger.info(f"[{self.symbol}] Stopping trading bot...")
        
        # Mark the bot as stopped
        self.is_running = False
        
        try:
            # Close websocket connection
            if hasattr(self, 'ws'):
                self.ws.disconnect()
            
            # Close any open positions based on configuration
            self.close_positions_on_shutdown()
            
            # Print final trading stats
            self.print_trading_summary()
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error during shutdown: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
    
    def close_positions_on_shutdown(self):
        """Close any open positions when shutting down"""
        try:
            # Get current position
            position = self.client.get_position_info(self.symbol)
            position_amt = float(position['positionAmt']) if position else 0
            
            if position_amt != 0:
                logger.info(f"[{self.symbol}] Closing position on shutdown: {position_amt}")
                self.telegram.send_message(f"🔒 [{self.symbol}] Closing positions on shutdown: {position_amt} {self.symbol}")
                
                side = "SELL" if position_amt > 0 else "BUY"
                self.client.place_market_order(self.symbol, side, abs(position_amt))
                
                # Cancel any open orders
                self.client.cancel_all_orders(self.symbol)
                
        except Exception as e:
            error_msg = f"[{self.symbol}] Error closing positions on shutdown: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
    
    def print_trading_summary(self):
        """Print a summary of trading performance"""
        try:
            # Get final account balance
            final_balance = self.client.get_account_balance()
            initial_balance = self.trading_stats['start_balance']
            
            # Calculate profit/loss
            pnl = final_balance - initial_balance
            pnl_percentage = (pnl / initial_balance) * 100 if initial_balance > 0 else 0
            
            summary_text = f"\n===== Trading Summary for {self.symbol} =====\n"
            summary_text += f"Strategy: {self.strategy_name}\n"
            summary_text += f"Initial Balance: {initial_balance:.2f} USDT\n"
            summary_text += f"Final Balance: {final_balance:.2f} USDT\n"
            summary_text += f"Profit/Loss: {pnl:.2f} USDT ({pnl_percentage:.2f}%)\n"
            summary_text += f"Total Trades: {self.trading_stats['total_trades']}\n"
            summary_text += f"Max Drawdown: {self.trading_stats['max_drawdown']:.2f} USDT\n"
            summary_text += "==========================\n"
            
            logger.info(summary_text)
            
            # Send summary to Telegram
            emoji = "📈" if pnl >= 0 else "📉"
            telegram_summary = f"{emoji} <b>Trading Summary for {self.symbol}</b>\n\n"
            telegram_summary += f"Strategy: {self.strategy_name}\n"
            telegram_summary += f"Initial Balance: {initial_balance:.2f} USDT\n"
            telegram_summary += f"Final Balance: {final_balance:.2f} USDT\n"
            telegram_summary += f"Profit/Loss: {pnl:.2f} USDT ({pnl_percentage:.2f}%)\n"
            telegram_summary += f"Total Trades: {self.trading_stats['total_trades']}\n"
            telegram_summary += f"Max Drawdown: {self.trading_stats['max_drawdown']:.2f} USDT"
            
            self.telegram.send_message(telegram_summary)
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error printing trading summary: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)

    def close_position(self, position, current_price):
        """Close an existing position"""
        try:
            position_amt = float(position['positionAmt'])
            if position_amt == 0:
                return
                
            symbol = position['symbol']
            side = "SELL" if position_amt > 0 else "BUY"  # Opposite side to close
            quantity = abs(position_amt)
            
            # Cancel existing orders first
            self.client.cancel_all_orders(symbol)
            
            # Execute the market order to close position
            order = self.client.place_market_order(symbol, side, quantity)
            
            if order:
                entry_price = float(position['entryPrice'])
                pnl = float(position['unRealizedProfit'])
                pnl_percentage = (pnl / (entry_price * quantity)) * 100 if entry_price * quantity != 0 else 0
                
                logger.info(f"[{symbol}] Closed {side} position of {quantity} at {current_price}, PnL: {pnl:.2f} USDT ({pnl_percentage:.2f}%)")
                
                # Send notification
                self.telegram.send_trade_closed_notification(
                    symbol, 
                    "BUY" if position_amt > 0 else "SELL",
                    quantity, 
                    entry_price, 
                    current_price, 
                    pnl, 
                    pnl_percentage
                )
                
                # Update trade statistics
                self.trading_stats['total_trades'] += 1
                self.daily_stats['total_trades'] += 1
                
                if pnl > 0:
                    self.trading_stats['winning_trades'] += 1
                    self.daily_stats['winning_trades'] += 1
                else:
                    self.trading_stats['losing_trades'] += 1
                    self.daily_stats['losing_trades'] += 1
                    
                # Update parent bot's last trade time if we have access to it
                try:
                    import inspect
                    for frame in inspect.stack():
                        if 'self' in frame[0].f_locals:
                            instance = frame[0].f_locals['self']
                            if isinstance(instance, MultiSymbolTradingBot):
                                instance.last_trade_time = datetime.now()
                                break
                except Exception:
                    pass
                
                return order
        except Exception as e:
            error_msg = f"[{self.symbol}] Error closing position: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
            return None
    
    def calculate_volatility_adjusted_position(self, symbol, price, base_risk_percentage):
        """Calculate position size adjusted for market volatility"""
        try:
            # Get recent price data to calculate volatility
            if self.klines_df is None or len(self.klines_df) < 20:
                return None
                
            # Calculate Average True Range (ATR) for volatility measurement
            high = self.klines_df['high'].values
            low = self.klines_df['low'].values
            close = self.klines_df['close'].values
            
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).T
            atr = np.mean(np.max(tr, axis=1)[-14:])  # 14-period ATR
            
            # Calculate volatility ratio (ATR as % of price)
            volatility_ratio = atr / price
            
            # Adjust risk based on volatility
            # Lower risk for high volatility, higher risk for low volatility
            volatility_adjusted_risk = base_risk_percentage
            
            if (volatility_ratio > 0.02):  # High volatility (>2%)
                volatility_adjusted_risk = max(base_risk_percentage * 0.5, 0.1)  # Reduce risk
            elif (volatility_ratio < 0.005):  # Low volatility (<0.5%)
                volatility_adjusted_risk = min(base_risk_percentage * 1.5, 2.0)  # Increase risk
                
            # Calculate new position size using adjusted risk
            balance = self.client.get_account_balance()
            risk_amount = balance * (volatility_adjusted_risk / 100)
            
            # Get leverage
            for pair in config.TRADING_PAIRS:
                if pair.get("symbol") == symbol:
                    leverage = pair.get("leverage", config.DEFAULT_LEVERAGE)
                    break
            else:
                leverage = config.DEFAULT_LEVERAGE
                
            position_size_in_usd = risk_amount * leverage
            quantity = position_size_in_usd / price
            
            # Round to appropriate precision
            symbol_info = self.client.client.get_symbol_info(symbol)
            quantity_precision = 3  # Default
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    quantity_precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
                    break
                    
            quantity = round(quantity, quantity_precision)
            
            logger.info(f"[{symbol}] Volatility-adjusted risk: {volatility_adjusted_risk:.2f}%, Position size: {quantity}")
            return quantity
            
        except Exception as e:
            logger.error(f"[{symbol}] Error calculating volatility-adjusted position: {e}")
            return None
    
    def set_trailing_stop_loss(self, symbol, side, quantity, activation_price, callback_rate=0.8):
        """Set a trailing stop loss order"""
        try:
            # Side here is the position side (BUY or SELL)
            # For trailing stop, we need the opposite side
            close_side = "SELL" if side == "BUY" else "BUY"
            
            # Calculate activation price (typically current market price)
            if not activation_price:
                ticker = self.client.client.futures_symbol_ticker(symbol=symbol)
                activation_price = float(ticker['price'])
            
            # Callback rate is the percentage distance from activation price (0.8 = 0.8%)
            params = {
                'symbol': symbol,
                'side': close_side,
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'quantity': quantity,
                'activationPrice': activation_price,
                'timeInForce': 'GTC'
            }
            
            # Place the trailing stop order
            order = self.client.client.futures_create_order(**params)
            logger.info(f"[{symbol}] Trailing stop loss set at {activation_price} with {callback_rate}% callback")
            
            # Send notification
            self.telegram.send_message(
                f"🔄 <b>Trailing Stop Set</b>\n"
                f"<b>Symbol:</b> {symbol}\n"
                f"<b>Position:</b> {'LONG' if side == 'BUY' else 'SHORT'}\n"
                f"<b>Activation Price:</b> {activation_price}\n"
                f"<b>Callback Rate:</b> {callback_rate}%"
            )
            
            return order
        except Exception as e:
            logger.error(f"[{symbol}] Error setting trailing stop loss: {e}")
            return None

    def get_trend_confirmation(self):
        """Get trend confirmation from multiple timeframes"""
        if self.klines_df is None or len(self.klines_df) < 50:
            return 0
            
        # Create a copy of the DataFrame to avoid modifying the original
        df = self.klines_df.copy()
        
        # Calculate EMAs for short, medium, and long timeframes
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Check if price is above/below all EMAs
        last_row = df.iloc[-1]
        close = last_row['close']
        
        # Strong uptrend: price > ema9 > ema21 > ema50
        if (close > last_row['ema9'] > last_row['ema21'] > last_row['ema50']):
            return 1  # Bullish confirmation
            
        # Strong downtrend: price < ema9 < ema21 < ema50
        elif (close < last_row['ema9'] < last_row['ema21'] < last_row['ema50']):
            return -1  # Bearish confirmation
            
        # Check if at least two EMAs confirm the trend
        if close > last_row['ema9'] and close > last_row['ema21']:
            return 0.5  # Moderate bullish
        elif close < last_row['ema9'] and close < last_row['ema21']:
            return -0.5  # Moderate bearish
            
        return 0  # No clear trend

    def analyze_market_condition(self):
        """Analyze market condition to adjust trading parameters"""
        if self.klines_df is None or len(self.klines_df) < 50:
            return "NORMAL"
            
        # Calculate recent volatility using ATR
        df = self.klines_df.copy()
        
        # Calculate Average True Range (ATR)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.vstack([tr1, tr2, tr3]).T
        atr = np.mean(np.max(tr, axis=1)[-14:])  # 14-period ATR
        
        # Calculate ATR as percentage of price
        current_price = df['close'].iloc[-1]
        atr_percentage = (atr / current_price) * 100
        
        # Calculate ADX for trend strength
        df['tr'] = np.max(np.vstack([tr1, tr2, tr3]).T, axis=1)
        df['tr'] = pd.Series(df['tr'], index=df.index[1:])
        
        # +DM and -DM
        df['plus_dm'] = np.zeros(len(df))
        df['minus_dm'] = np.zeros(len(df))
        
        # Calculate +DM and -DM
        for i in range(1, len(df)):
            df['plus_dm'].iloc[i] = max(0, df['high'].iloc[i] - df['high'].iloc[i-1])
            df['minus_dm'].iloc[i] = max(0, df['low'].iloc[i-1] - df['low'].iloc[i])
            
            # If +DM > -DM, then -DM = 0
            if df['plus_dm'].iloc[i] > df['minus_dm'].iloc[i]:
                df['minus_dm'].iloc[i] = 0
            # If -DM > +DM, then +DM = 0
            elif df['minus_dm'].iloc[i] > df['plus_dm'].iloc[i]:
                df['plus_dm'].iloc[i] = 0
        
        # Calculate smoothed +DM, -DM and TR
        period = 14
        df['smoothed_tr'] = df['tr'].rolling(window=period).sum()
        df['smoothed_plus_dm'] = df['plus_dm'].rolling(window=period).sum()
        df['smoothed_minus_dm'] = df['minus_dm'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        df['+di'] = 100 * df['smoothed_plus_dm'] / df['smoothed_tr']
        df['-di'] = 100 * df['smoothed_minus_dm'] / df['smoothed_tr']
        
        # Calculate directional movement index (DX)
        df['dx'] = 100 * np.abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Get most recent ADX value
        recent_adx = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
        
        # Determine market condition
        if atr_percentage > 3.0:  # High volatility
            return "VOLATILE"
        elif recent_adx > 25:  # Strong trend
            return "TRENDING"
        else:
            return "NORMAL"

    def get_real_time_price(self):
        """Get the latest real-time price from websocket data"""
        # First try the WebSocket's cached price
        if hasattr(self, 'ws') and self.ws:
            price = self.ws.get_latest_price()
            if price > 0:
                return price
                
        # Fall back to the latest known bid/ask
        if self.latest_ask > 0 and self.latest_bid > 0:
            return (self.latest_ask + self.latest_bid) / 2
            
        # If all else fails, get a fresh quote via REST API
        try:
            ticker = self.client.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"[{self.symbol}] Error getting real-time price: {e}")
            return 0

    def update_latest_data_via_rest(self):
        """Fetch latest market data using REST API as fallback when WebSocket is down"""
        try:
            # Get current symbol ticker
            ticker = self.client.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            logger.info(f"[{self.symbol}] Updated price via REST API: {current_price}")
            
            # Fetch latest completed candle
            interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            interval = interval_map.get(self.timeframe, '1h')
            
            # Get just the last 2 candles
            klines = self.client.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=2
            )
            
            if len(klines) > 0:
                # Add to existing data if it's a new candle
                if len(self.klines_data) == 0 or klines[-1][0] > self.klines_data[-1][0]:
                    self.add_kline_to_data(klines[-1])
                    logger.info(f"[{self.symbol}] Added new candle via REST API")
                    self.update_strategy()
                    
        except Exception as e:
            logger.error(f"[{self.symbol}] Error in REST API update: {e}")


class MultiSymbolTradingBot:
    """Main class for managing multiple trading bots for different symbols"""
    
    def __init__(self):
        self.client = BinanceClient()
        self.telegram = TelegramNotifier()
        self.bots = {}
        self.threads = {}
        self.is_running = False
        
        # Initialize account status notification scheduler
        self.last_status_time = datetime.now()
        
        # Add startup time for uptime calculation
        self.startup_time = datetime.now()
        
        # Add last trade time tracking
        self.last_trade_time = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("Multi-symbol trading bot initialized")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, closing all trading bots...")
        self.telegram.send_message("⚠️ <b>Multi-symbol bot shutting down</b>")
        self.stop()
        sys.exit(0)
        
    def send_account_status_notification(self):
        """Send a detailed account status notification"""
        try:
            # Get detailed account information
            account_info = self.client.get_detailed_account_info()
            positions = self.client.get_all_open_positions()
            
            # Add fallback handling in case we still have issues
            if not account_info:
                logger.warning("Failed to get account info, using minimal data for notification")
                account_info = {
                    'totalWalletBalance': '1000',
                    'totalUnrealizedProfit': '0',
                    'availableBalance': '1000'
                }
            
            # Send the notification with detailed account info
            self.telegram.send_detailed_account_status(account_info, positions)
            
            logger.info("Sent detailed account status notification")
        except Exception as e:
            logger.error(f"Error sending account status: {e}")
            # Send a simplified notification instead
            try:
                balance = self.client.get_account_balance()
                self.telegram.send_message(f"📊 Account Balance: {balance} USDT\n⚠️ Detailed status unavailable")
            except Exception:
                self.telegram.send_message("⚠️ Unable to fetch account information")
    
    def check_api_and_websocket_status(self):
        """Check if API and WebSocket connections are working"""
        api_status = False
        websocket_status = False
        
        try:
            # Check API connection
            try:
                time_data = self.client.client.get_server_time()
                if time_data and 'serverTime' in time_data:
                    api_status = True
                    logger.info("API connection working")
            except Exception as e:
                logger.error(f"API connection failed: {e}")
            
            # Give WebSocket connections more time to establish
            time.sleep(5)
            
            # Check WebSocket connection for at least one bot
            if self.bots:
                websocket_checks = []
                for symbol, bot in self.bots.items():
                    if hasattr(bot, 'ws') and bot.ws:
                        is_alive = bot.ws.is_alive()
                        is_connected = getattr(bot.ws, 'is_connected', False)
                        websocket_checks.append((symbol, is_alive, is_connected))
                        
                        if is_alive and is_connected:
                            websocket_status = True
                            logger.info(f"WebSocket connection working for {symbol}")
                            break
                
                if not websocket_status:
                    logger.warning(f"WebSocket connection check failed. Status: {websocket_checks}")
                    
                    # Try to directly check a WebSocket connection
                    try:
                        self._verify_direct_websocket_connection()
                        websocket_status = True
                    except Exception as ws_e:
                        logger.error(f"Direct WebSocket check failed: {ws_e}")
            
            return api_status, websocket_status
        except Exception as e:
            logger.error(f"Error checking connections: {e}")
            return api_status, websocket_status
    
    def _verify_direct_websocket_connection(self):
        """Verify WebSocket connection directly"""
        import websocket as direct_ws
        import json
        
        logger.info("Performing direct WebSocket connection check")
        
        connection_success = threading.Event()
        message_received = threading.Event()
        
        def on_message(ws, message):
            logger.info(f"Direct WebSocket test - Received message: {message[:100]}...")
            message_received.set()
            
        def on_open(ws):
            logger.info("Direct WebSocket test - Connection opened")
            connection_success.set()
            
            # Subscribe to a simple stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@bookTicker"],
                "id": int(time.time())
            }
            ws.send(json.dumps(subscribe_msg))
        
        def on_error(ws, error):
            logger.error(f"Direct WebSocket test - Error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Direct WebSocket test - Connection closed: {close_status_code}, {close_msg}")
        
        # Create and connect WebSocket
        ws_url = "wss://fstream.binancefuture.com/ws"
        if config.TEST_MODE:
            ws_url = "wss://stream.binancefuture.com/ws"
            
        ws = direct_ws.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in a thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection
        if not connection_success.wait(timeout=10):
            raise Exception("Failed to connect to WebSocket directly")
            
        # Wait for message
        if not message_received.wait(timeout=10):
            raise Exception("No message received from direct WebSocket connection")
            
        # Close WebSocket
        ws.close()
        logger.info("Direct WebSocket connection test passed")
        return True
    
    def start(self):
        """Start trading bots for all configured symbols"""
        try:
            if self.is_running:
                logger.warning("Trading bots are already running")
                return
                
            logger.info("Starting multi-symbol trading bot...")
            
            # Send initial status notification
            self.telegram.send_message("🚀 <b>Trading Bot Starting</b>\nInitializing systems and connections...")
            
            # Check API connection before starting (first step)
            try:
                time_data = self.client.client.get_server_time()
                if time_data and 'serverTime' in time_data:
                    api_status = True
                    logger.info("API connection working")
                else:
                    api_status = False
                    logger.warning("API connection check returned unexpected response")
            except Exception as e:
                api_status = False
                logger.error(f"API connection check failed: {e}")
            
            if not api_status:
                self.telegram.send_error_notification("Failed to connect to API. Check your network connection and API keys.")
                return
            
            # Create and start a bot for each configured trading pair
            for trading_pair in config.TRADING_PAIRS:
                symbol = trading_pair.get("symbol")
                
                if not symbol:
                    continue
                
                # Create a new bot instance for this symbol
                self.bots[symbol] = TradingBot(trading_pair)
                
                # Start the bot in a separate thread
                thread = threading.Thread(target=self.bots[symbol].start, name=f"Bot-{symbol}")
                thread.daemon = True
                self.threads[symbol] = thread
                thread.start()
                
                logger.info(f"Started trading bot for {symbol}")
                
                # Small delay to prevent API rate limit issues
                time.sleep(3)
            
            self.is_running = True
            
            # Allow time for WebSocket connections to establish
            logger.info("Waiting for WebSocket connections to establish...")
            time.sleep(10)
            
            # Check API and WebSocket connections after bots have started
            api_status, websocket_status = self.check_api_and_websocket_status()
            
            # Send startup notification
            if api_status and websocket_status:
                self.telegram.send_startup_confirmation()
                logger.info("Sent startup confirmation: All systems operational")
            else:
                status_msg = f"Startup with warnings - API: {'OK' if api_status else 'ERROR'}, WebSocket: {'OK' if websocket_status else 'ERROR'}"
                self.telegram.send_system_status_notification(api_status, websocket_status)
                logger.warning(status_msg)
                
                # If WebSocket is not connected, try to help users
                if not websocket_status:
                    help_msg = ("⚠️ <b>WebSocket Connection Issue</b>\n\n"
                               "The bot will continue to operate using REST API as fallback.\n\n"
                               "Possible solutions:\n"
                               "1. Check your network connection\n"
                               "2. Make sure port 443 is not blocked\n"
                               "3. Restart the bot\n"
                               "4. Verify Binance Futures API status")
                    self.telegram.send_message(help_msg)
            
            # Send initial account status
            self.send_account_status_notification()
            
            # Set up schedules for regular notifications
            schedule.every(3).hours.do(self.send_account_status_notification)
            schedule.every(5).hours.do(self.send_periodic_status_notification)
            
            # Keep the main thread alive
            while self.is_running:
                # Check if all threads are still alive
                for symbol, thread in list(self.threads.items()):
                    if not thread.is_alive():
                        logger.warning(f"Thread for {symbol} died, restarting...")
                        self.telegram.send_message(f"⚠️ <b>Bot for {symbol} crashed, restarting...</b>")
                        
                        # Restart the bot
                        thread = threading.Thread(target=self.bots[symbol].start, name=f"Bot-{symbol}")
                        thread.daemon = True
                        self.threads[symbol] = thread
                        thread.start()
                
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Check if it's time to send account status (every 3 hours)
                current_time = datetime.now()
                time_since_last_status = (current_time - self.last_status_time).total_seconds()
                
                if time_since_last_status > 10800:  # 3 hours in seconds
                    self.send_account_status_notification()
                    self.last_status_time = current_time
                
                time.sleep(30)  # Check thread status every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.telegram.send_message("🛑 Multi-symbol trading bot stopped by user")
            self.stop()
        except Exception as e:
            error_msg = f"Error in multi-symbol trading bot: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
            self.stop()
    
    def stop(self):
        """Stop all trading bots"""
        if not self.is_running:
            return
            
        logger.info("Stopping all trading bots...")
        self.is_running = False
        
        # Final account status notification
        self.send_account_status_notification()
        
        # Stop each bot
        for symbol, bot in self.bots.items():
            try:
                logger.info(f"Stopping bot for {symbol}...")
                bot.stop()
            except Exception as e:
                logger.error(f"Error stopping bot for {symbol}: {e}")
        
        # Wait for all threads to finish (with a timeout)
        for symbol, thread in self.threads.items():
            try:
                thread.join(timeout=10)
            except Exception as e:
                logger.error(f"Error joining thread for {symbol}: {e}")
        
        self.telegram.send_message("🛑 <b>All trading bots stopped</b>")
        logger.info("All trading bots stopped")

    def send_periodic_status_notification(self):
        """Send a periodic status notification"""
        try:
            # Calculate uptime
            uptime = datetime.now() - self.startup_time
            uptime_hours = uptime.total_seconds() / 3600
            
            # Get active symbols
            active_symbols = [symbol for symbol, bot in self.bots.items() if bot.is_running]
            
            # Get account balance
            balance = self.client.get_account_balance()
            
            # Get position information
            positions = self.client.get_all_open_positions()
            
            # Send the notification
            self.telegram.send_periodic_status_notification(
                uptime_hours=uptime_hours,
                active_symbols=active_symbols,
                last_trade_time=self.last_trade_time,
                balance=balance,
                positions=positions
            )
            
            logger.info(f"Sent periodic status notification. Uptime: {uptime_hours:.1f} hours")
        except Exception as e:
            logger.error(f"Error sending periodic status: {e}")
            # Send a basic notification instead
            try:
                uptime = datetime.now() - self.startup_time
                uptime_hours = uptime.total_seconds() / 3600
                self.telegram.send_message(f"📊 <b>Bot Status Update</b>\nUptime: {uptime_hours:.1f} hours\nBot is running")
            except:
                pass


if __name__ == "__main__":
    try:
        # Create and start the multi-symbol trading bot
        bot_manager = MultiSymbolTradingBot()
        bot_manager.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)