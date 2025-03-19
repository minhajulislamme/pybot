import logging
import requests
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Class for sending Telegram notifications"""
    
    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.notification_enabled = config.NOTIFICATIONS_ENABLED
        
        if self.notification_enabled:
            logger.info("Telegram notifications enabled")
            self.send_message("🤖 <b>Trading Bot Started</b> - Monitoring market conditions...")
        else:
            logger.info("Telegram notifications disabled")
            
    def send_message(self, message):
        """Send a message to the Telegram chat"""
        if not self.notification_enabled:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def send_error_notification(self, error_message):
        """Send an error notification"""
        message = f"❌ <b>ERROR</b>: {error_message}"
        self.send_message(message)
    
    def send_signal_notification(self, symbol, signal_type, price):
        """Send a trading signal notification"""
        emoji = "🔴" if signal_type == "SELL" else "🟢"
        message = f"{emoji} <b>Signal: {signal_type}</b>\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Price:</b> {price}\n" \
                  f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)

    def send_trade_notification(self, symbol, side, amount, price):
        """Send a notification when a trade is executed"""
        emoji = "🟢 BUY" if side == "BUY" else "🔴 SELL"
        message = f"<b>{emoji} Order Executed</b>\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Amount:</b> {amount}\n" \
                  f"<b>Price:</b> {price}\n" \
                  f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)
    
    def send_trade_closed_notification(self, symbol, side, amount, entry_price, exit_price, pnl, pnl_percentage):
        """Send a notification when a position is closed"""
        profit_loss = "PROFIT ✅" if pnl >= 0 else "LOSS ❌"
        side_emoji = "🟢 LONG" if side == "BUY" else "🔴 SHORT"
        
        # Format PnL with 2 decimal places
        pnl_formatted = f"{pnl:.2f}"
        pnl_percentage_formatted = f"{pnl_percentage:.2f}"
        
        message = f"<b>Position Closed - {profit_loss}</b>\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Position:</b> {side_emoji}\n" \
                  f"<b>Amount:</b> {amount}\n" \
                  f"<b>Entry:</b> {entry_price}\n" \
                  f"<b>Exit:</b> {exit_price}\n" \
                  f"<b>P&L:</b> {pnl_formatted} USDT ({pnl_percentage_formatted}%)\n" \
                  f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)
        
    def send_sl_tp_notification(self, symbol, side, sl_price, tp_price):
        """Send notification about stop loss and take profit levels"""
        side_text = "LONG" if side == "BUY" else "SHORT"
        message = f"<b>SL/TP Set for {side_text} Position</b>\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Stop Loss:</b> {sl_price}\n" \
                  f"<b>Take Profit:</b> {tp_price}\n" \
                  f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)
        
    def send_balance_update(self, balance, pnl, pnl_percentage):
        """Send a notification about the current balance"""
        emoji = "📈" if pnl >= 0 else "📉"
        
        # Format values with appropriate decimal places
        balance_formatted = f"{balance:.2f}"
        pnl_formatted = f"{pnl:.2f}"
        pnl_percentage_formatted = f"{pnl_percentage:.2f}"
        
        message = f"{emoji} <b>Balance Update</b>\n" \
                  f"<b>Current Balance:</b> {balance_formatted} USDT\n" \
                  f"<b>P&L:</b> {pnl_formatted} USDT ({pnl_percentage_formatted}%)\n" \
                  f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)
        
    def send_detailed_account_status(self, account_info, positions):
        """Send a detailed account status with all positions"""
        # Get account metrics
        total_wallet_balance = float(account_info.get('totalWalletBalance', 0))
        total_unrealized_profit = float(account_info.get('totalUnrealizedProfit', 0))
        available_balance = float(account_info.get('availableBalance', 0))
        
        # Format account metrics
        wallet_formatted = f"{total_wallet_balance:.2f}"
        unrealized_formatted = f"{total_unrealized_profit:.2f}"
        available_formatted = f"{available_balance:.2f}"
        
        # Determine emoji based on unrealized profit
        account_emoji = "📈" if total_unrealized_profit >= 0 else "📉"
        profit_emoji = "✅" if total_unrealized_profit >= 0 else "❌"
        
        # Create message
        message = f"{account_emoji} <b>ACCOUNT STATUS</b>\n\n" \
                 f"<b>Wallet Balance:</b> {wallet_formatted} USDT\n" \
                 f"<b>Unrealized P&L:</b> {unrealized_formatted} USDT {profit_emoji}\n" \
                 f"<b>Available Balance:</b> {available_formatted} USDT\n"
        
        # Add position information if any positions exist
        if positions and len(positions) > 0:
            message += "\n<b>OPEN POSITIONS:</b>\n"
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                position_amt = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_profit = float(pos.get('unrealizedProfit', 0))
                
                # Skip positions with zero amount
                if position_amt == 0:
                    continue
                
                # Determine position side and emoji
                if position_amt > 0:
                    side_emoji = "🟢"
                    side = "LONG"
                else:
                    side_emoji = "🔴"
                    side = "SHORT"
                    position_amt = abs(position_amt)
                
                # Calculate PnL percentage
                pnl_percentage = 0
                if entry_price > 0:
                    if side == "LONG":
                        pnl_percentage = ((mark_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percentage = ((entry_price - mark_price) / entry_price) * 100
                
                # Format values
                position_formatted = f"{position_amt:.4f}"
                entry_formatted = f"{entry_price:.2f}"
                mark_formatted = f"{mark_price:.2f}"
                profit_formatted = f"{unrealized_profit:.2f}"
                percentage_formatted = f"{pnl_percentage:.2f}"
                
                # Determine profit emoji
                pos_profit_emoji = "✅" if unrealized_profit >= 0 else "❌"
                
                message += f"\n{side_emoji} <b>{symbol}</b> - {side}\n" \
                          f"  Amount: {position_formatted}\n" \
                          f"  Entry: {entry_formatted}\n" \
                          f"  Current: {mark_formatted}\n" \
                          f"  P&L: {profit_formatted} USDT ({percentage_formatted}%) {pos_profit_emoji}\n"
        else:
            message += "\n<b>No open positions</b>"
        
        message += f"\n<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_message(message)
        
    def send_daily_summary(self, stats):
        """Send a daily trading summary"""
        # Get key statistics
        total_trades = stats.get('total_trades', 0)
        winning_trades = stats.get('winning_trades', 0)
        losing_trades = stats.get('losing_trades', 0)
        start_balance = stats.get('start_balance', 0)
        current_balance = stats.get('current_balance', 0)
        total_profit_loss = stats.get('total_profit_loss', 0)
        max_drawdown = stats.get('max_drawdown', 0)
        
        # Calculate win rate and other metrics
        win_rate = 0
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            
        pnl_percentage = 0
        if start_balance > 0:
            pnl_percentage = (total_profit_loss / start_balance) * 100
            
        drawdown_percentage = 0
        if start_balance > 0:
            drawdown_percentage = (max_drawdown / start_balance) * 100
        
        # Format values
        start_formatted = f"{start_balance:.2f}"
        current_formatted = f"{current_balance:.2f}"
        profit_formatted = f"{total_profit_loss:.2f}"
        pnl_percentage_formatted = f"{pnl_percentage:.2f}"
        win_rate_formatted = f"{win_rate:.1f}"
        drawdown_formatted = f"{max_drawdown:.2f}"
        drawdown_percentage_formatted = f"{drawdown_percentage:.2f}"
        
        # Determine emoji
        summary_emoji = "📈" if total_profit_loss >= 0 else "📉"
        profit_emoji = "✅" if total_profit_loss >= 0 else "❌"
        
        # Create message
        message = f"{summary_emoji} <b>DAILY TRADING SUMMARY</b>\n\n" \
                 f"<b>Performance:</b>\n" \
                 f"  Start Balance: {start_formatted} USDT\n" \
                 f"  Current Balance: {current_formatted} USDT\n" \
                 f"  P&L: {profit_formatted} USDT ({pnl_percentage_formatted}%) {profit_emoji}\n" \
                 f"  Max Drawdown: {drawdown_formatted} USDT ({drawdown_percentage_formatted}%)\n\n" \
                 f"<b>Trading Activity:</b>\n" \
                 f"  Total Trades: {total_trades}\n" \
                 f"  Winning Trades: {winning_trades}\n" \
                 f"  Losing Trades: {losing_trades}\n" \
                 f"  Win Rate: {win_rate_formatted}%\n\n" \
                 f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}"
                 
        self.send_message(message)