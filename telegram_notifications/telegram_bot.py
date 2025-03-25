"""
Telegram Bot for sending trading notifications and updates.
"""

import logging
import asyncio
import telegram
from datetime import datetime
from config.config import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Class for sending trading notifications to Telegram."""
    
    def __init__(self, token=None, chat_id=None):
        """
        Initialize Telegram notifier.
        
        Args:
            token (str, optional): Telegram bot token
            chat_id (str, optional): Telegram chat ID
        """
        self.token = token or config.telegram_token
        self.chat_id = chat_id or config.telegram_chat_id
        self.bot = None
        self.enabled = config.telegram_enabled
        self.event_loop = None
        self.initialize_bot()
        
    def initialize_bot(self):
        """Initialize the Telegram bot."""
        if not self.enabled:
            logger.info("Telegram notifications are disabled")
            return
            
        if not self.token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not provided, notifications disabled")
            self.enabled = False
            return
            
        try:
            self.bot = telegram.Bot(token=self.token)
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            logger.info("Telegram bot initialized")
            # Send a startup message
            self.send_message_sync("🤖 *Binance Futures Trading Bot started*\n"
                               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    async def send_message(self, message):
        """
        Send a message to the Telegram chat.
        
        Args:
            message (str): Message text to send
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.enabled or not self.bot:
            return False
            
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
            
    def send_message_sync(self, message):
        """
        Synchronous wrapper for send_message.
        
        Args:
            message (str): Message text to send
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.event_loop:
            try:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
            except Exception as e:
                logger.error(f"Failed to create event loop: {e}")
                return False

        try:
            return self.event_loop.run_until_complete(self.send_message(message))
        except RuntimeError as e:
            # If event loop is closed, create a new one
            if "Event loop is closed" in str(e):
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
                return self.event_loop.run_until_complete(self.send_message(message))
            raise
    
    def send_trade_notification(self, trade_type, symbol, side, quantity, price=None, pnl=None):
        """
        Send a notification about a trade.
        
        Args:
            trade_type (str): Type of trade ('entry', 'exit', 'stop_loss', 'take_profit')
            symbol (str): Trading symbol
            side (str): Trade side ('BUY' or 'SELL')
            quantity (float): Trade quantity
            price (float, optional): Trade price
            pnl (float, optional): Profit/loss for exit trades
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        if not self.enabled:
            return False
            
        emoji_map = {
            'entry': '🔵' if side == 'BUY' else '🔴',
            'exit': '🟢' if side == 'SELL' and pnl and pnl > 0 else '🔴',
            'stop_loss': '⛔',
            'take_profit': '💰'
        }
        
        emoji = emoji_map.get(trade_type, '🔄')
        
        message = f"{emoji} *{trade_type.upper()}*: {symbol}\n"
        message += f"Side: {side}\n"
        message += f"Quantity: {quantity}\n"
        
        if price:
            message += f"Price: {price}\n"
            
        if pnl is not None:
            message += f"PnL: {pnl:.2f} USDT "
            message += f"({'+'if pnl > 0 else ''}{pnl/price/quantity*100:.2f}%)\n"
            
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message_sync(message)
    
    def send_account_update(self, balance, open_positions=None):
        """
        Send an account update notification.
        
        Args:
            balance (float): Current account balance
            open_positions (list, optional): List of open positions
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        if not self.enabled:
            return False
            
        message = f"📊 *ACCOUNT UPDATE*\n"
        message += f"Balance: {balance:.2f} USDT\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if open_positions:
            message += "*Open Positions:*\n"
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                amount = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                pnl = float(pos.get('unrealizedProfit', 0))
                side = "LONG" if amount > 0 else "SHORT"
                
                message += f"- {symbol} ({side}): {abs(amount)} @ {entry_price}\n"
                message += f"  Current: {mark_price} | PnL: {pnl:.2f} USDT\n"
        else:
            message += "No open positions."
            
        return self.send_message_sync(message)
    
    def send_error(self, error_type, message):
        """
        Send an error notification.
        
        Args:
            error_type (str): Type of error
            message (str): Error message
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        if not self.enabled:
            return False
            
        error_message = f"⚠️ *ERROR: {error_type}*\n"
        error_message += f"{message}\n"
        error_message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message_sync(error_message)
    
    def send_strategy_signal(self, strategy_name, symbol, signal, details=None):
        """
        Send a notification about a strategy signal.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            signal (int): Signal value (1 for buy, -1 for sell, 0 for neutral)
            details (str, optional): Additional details about the signal
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        if not self.enabled:
            return False
            
        signal_text = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
        emoji = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        
        message = f"{emoji} *{strategy_name} SIGNAL*: {signal_text}\n"
        message += f"Symbol: {symbol}\n"
        
        if details:
            message += f"Details: {details}\n"
            
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message_sync(message)