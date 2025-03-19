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
                    self.ws = BinanceWebsocket(self.symbol, self.on_websocket_message, self.on_websocket_close)
                    self.ws.connect()
                    
                    # Subscribe to additional data streams
                    self.ws.subscribe_kline(self.symbol, self.timeframe)
                    self.ws.subscribe_book_ticker(self.symbol)
                    self.ws.subscribe_mark_price(self.symbol)
                    logger.info(f"[{self.symbol}] WebSocket connections initialized")
                    
                    return  # Success - exit the retry loop
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"[{self.symbol}] WebSocket connection attempt {retry_count} failed: {e}")
                    if retry_count < max_retries:
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise  # Re-raise the exception after max retries
            
        except Exception as e:
            error_msg = f"[{self.symbol}] Error initializing websocket after {max_retries} attempts: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
            raise
    
    def on_websocket_message(self, message):
        """Process incoming websocket message"""
        try:
            # Check if it's a kline/candlestick message
            if 'k' in message:
                self.process_kline_message(message)
            
            # Check if it's a book ticker message (best bid/ask)
            elif 'b' in message and 'a' in message and 's' in message:
                self.process_book_ticker_message(message)
                
            # Check if it's a mark price message
            elif 'e' in message and message['e'] == 'markPriceUpdate':
                self.process_mark_price_message(message)
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error processing websocket message: {e}")
    
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
        kline = message['k']
        
        # Only process if the candle is closed
        if kline['x']:
            logger.info(f"[{self.symbol}] Closed candle: {kline['t']} - Open: {kline['o']}, Close: {kline['c']}")
            
            # Add to klines data
            with self.lock:
                self.add_kline_to_data(kline)
                self.update_strategy()
        
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
        # Store the latest bid/ask prices
        self.latest_bid = float(message['b'])
        self.latest_ask = float(message['a'])
    
    def process_mark_price_message(self, message):
        """Process mark price updates"""
        # Store the latest mark price
        self.latest_mark_price = float(message['p'])
    
    def update_strategy(self):
        """Update strategy with latest data and process signals"""
        if self.klines_df is None or len(self.klines_df) < 50:  # Using a default minimum value
            logger.info(f"[{self.symbol}] Not enough data for strategy calculations yet")
            return
        
        # Calculate signals
        signals_df = self.strategy.calculate_signals(self.klines_df)
        
        # Check the latest signal
        latest_signal = signals_df.iloc[-1]['signal']
        
        if latest_signal != 0:
            # Process the trading signal
            self.process_trading_signal(latest_signal, signals_df)
    
    def process_trading_signal(self, signal, data):
        """Process a trading signal and execute trades accordingly"""
        logger.info(f"[{self.symbol}] Processing signal: {signal}")
        
        try:
            # Get current position info
            position = self.client.get_position_info(self.symbol)
            current_position_size = float(position['positionAmt']) if position else 0
            
            # Current price (use latest ask for buys, bid for sells)
            current_price = self.latest_ask if signal > 0 else self.latest_bid
            
            # Position side
            position_side = "BUY" if signal > 0 else "SELL"
            
            # Send signal notification
            self.telegram.send_signal_notification(self.symbol, "BUY" if signal > 0 else "SELL", current_price)
            
            if signal == 1:  # Buy signal
                if current_position_size <= 0:  # No position or short position
                    # Close any existing short position
                    if current_position_size < 0:
                        logger.info(f"[{self.symbol}] Closing existing short position: {current_position_size}")
                        
                        # Track entry price for P&L calculation
                        entry_price = float(position['entryPrice']) if position else 0
                        
                        # Close short position
                        close_order = self.client.place_market_order(self.symbol, "BUY", abs(current_position_size))
                        self.client.cancel_all_orders(self.symbol)  # Cancel existing SL/TP orders
                        
                        # Calculate and report P&L
                        pnl = (entry_price - current_price) * abs(current_position_size) * self.leverage
                        pnl_percentage = ((entry_price - current_price) / entry_price) * 100 * self.leverage
                        
                        # Update trading stats based on profitability
                        if pnl >= 0:
                            self.trading_stats['winning_trades'] += 1
                            self.daily_stats['winning_trades'] += 1
                        else:
                            self.trading_stats['losing_trades'] += 1
                            self.daily_stats['losing_trades'] += 1
                        
                        # Send notification about closed position
                        self.telegram.send_trade_closed_notification(
                            self.symbol, "SELL", abs(current_position_size), 
                            entry_price, current_price, pnl, pnl_percentage
                        )
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(current_price)
                    
                    # Open long position
                    logger.info(f"[{self.symbol}] Opening long position: {position_size} at ~{current_price}")
                    order = self.client.place_market_order(self.symbol, "BUY", position_size)
                    
                    # Notify about new trade
                    self.telegram.send_trade_notification(self.symbol, "BUY", position_size, current_price)
                    
                    # Set stop loss and take profit
                    sl_tp_orders = self.risk_manager.set_stop_loss_and_take_profit(current_price, "BUY", position_size)
                    
                    # Calculate SL and TP prices
                    sl_price = self.risk_manager.calculate_stop_loss_price(current_price, "BUY")
                    tp_price = self.risk_manager.calculate_take_profit_price(current_price, "BUY")
                    
                    # Send notification about SL/TP
                    self.telegram.send_sl_tp_notification(self.symbol, "BUY", sl_price, tp_price)
                    
                    # Update trading stats
                    self.trading_stats['total_trades'] += 1
                    self.daily_stats['total_trades'] += 1
                    
            elif signal == -1:  # Sell signal
                if current_position_size >= 0:  # No position or long position
                    # Close any existing long position
                    if current_position_size > 0:
                        logger.info(f"[{self.symbol}] Closing existing long position: {current_position_size}")
                        
                        # Track entry price for P&L calculation
                        entry_price = float(position['entryPrice']) if position else 0
                        
                        # Close long position
                        close_order = self.client.place_market_order(self.symbol, "SELL", current_position_size)
                        self.client.cancel_all_orders(self.symbol)  # Cancel existing SL/TP orders
                        
                        # Calculate and report P&L
                        pnl = (current_price - entry_price) * current_position_size * self.leverage
                        pnl_percentage = ((current_price - entry_price) / entry_price) * 100 * self.leverage
                        
                        # Update trading stats based on profitability
                        if pnl >= 0:
                            self.trading_stats['winning_trades'] += 1
                            self.daily_stats['winning_trades'] += 1
                        else:
                            self.trading_stats['losing_trades'] += 1
                            self.daily_stats['losing_trades'] += 1
                        
                        # Send notification about closed position
                        self.telegram.send_trade_closed_notification(
                            self.symbol, "BUY", current_position_size, 
                            entry_price, current_price, pnl, pnl_percentage
                        )
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(current_price)
                    
                    # Open short position
                    logger.info(f"[{self.symbol}] Opening short position: {position_size} at ~{current_price}")
                    order = self.client.place_market_order(self.symbol, "SELL", position_size)
                    
                    # Notify about new trade
                    self.telegram.send_trade_notification(self.symbol, "SELL", position_size, current_price)
                    
                    # Set stop loss and take profit
                    sl_tp_orders = self.risk_manager.set_stop_loss_and_take_profit(current_price, "SELL", position_size)
                    
                    # Calculate SL and TP prices
                    sl_price = self.risk_manager.calculate_stop_loss_price(current_price, "SELL")
                    tp_price = self.risk_manager.calculate_take_profit_price(current_price, "SELL")
                    
                    # Send notification about SL/TP
                    self.telegram.send_sl_tp_notification(self.symbol, "SELL", sl_price, tp_price)
                    
                    # Update trading stats
                    self.trading_stats['total_trades'] += 1
                    self.daily_stats['total_trades'] += 1
            
            # Update trading stats after trade
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
            error_msg = f"[{self.symbol}] Error fetching initial data: {e}"
            logger.error(error_msg)
            self.telegram.send_error_notification(error_msg)
            raise
    
    def start(self):
        """Start the trading bot for a specific symbol"""
        try:
            logger.info(f"[{self.symbol}] Starting trading bot...")
            
            # Fetch initial historical data
            self.fetch_initial_data()
            
            # Initialize websocket connection
            self.initialize_websocket()
            
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
    
    def start(self):
        """Start trading bots for all configured symbols"""
        try:
            if self.is_running:
                logger.warning("Trading bots are already running")
                return
                
            logger.info("Starting multi-symbol trading bot...")
            self.telegram.send_message("🚀 <b>Multi-Symbol Trading Bot Starting</b>")
            
            # Send initial account status
            self.send_account_status_notification()
            
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
                time.sleep(2)
            
            self.is_running = True
            
            # Set up schedule for regular account status updates (every 3 hours)
            schedule.every(3).hours.do(self.send_account_status_notification)
            
            # Keep the main thread alive
            while self.is_running:
                # Check if all threads are still alive
                for symbol, thread in self.threads.items():
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
                if (current_time - self.last_status_time).total_seconds() > 10800:  # 3 hours in seconds
                    self.send_account_status_notification()
                    self.last_status_time = current_time
                
                time.sleep(60)  # Check thread status every minute
                
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


if __name__ == "__main__":
    try:
        # Create and start the multi-symbol trading bot
        bot_manager = MultiSymbolTradingBot()
        bot_manager.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)