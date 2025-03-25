"""
Test script to execute a trade on Binance testnet and verify Telegram notifications.
"""

import logging
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from config.config import config
from data_fetchers.binance_client import BinanceClient
from telegram_notifications.telegram_bot import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_telegram_notification():
    """Test that Telegram notifications are working correctly"""
    telegram = TelegramNotifier(
        token=config.telegram_token,
        chat_id=config.telegram_chat_id
    )
    
    # Send test message
    result = telegram.send_message_sync("🧪 *TEST NOTIFICATION*\nTelegram notifications are now being tested.")
    logger.info(f"Telegram test notification result: {result}")
    return telegram, result

def test_trade(telegram):
    """Execute a test trade on Binance testnet"""
    try:
        # Initialize client with testnet credentials
        client = BinanceClient(
            api_key=config.testnet_api_key,
            api_secret=config.testnet_api_secret,
            testnet=True
        )
        
        # Check account balance
        balance = client.get_account_balance()
        logger.info(f"Current USDT balance: {balance}")
        
        # Get current price of BTC
        symbol = "BTCUSDT"  # This is already set as default in BinanceClient
        price = client.get_ticker_price()
        logger.info(f"Current {symbol} price: {price}")
        
        # Send notification about the test trade we're going to execute
        telegram.send_message_sync(f"🔄 *TEST TRADE INITIATED*\nSymbol: {symbol}\nPrice: ${price:.2f}\nBalance: ${balance:.2f}")
        
        # Calculate minimum position size to meet 100 USDT requirement
        min_notional = 100  # Binance requires at least 100 USDT
        position_size = min_notional / price
        position_size = round(position_size + 0.0005, 3)  # Add a small buffer and round to 3 decimal places
        logger.info(f"Calculated position size: {position_size} BTC (value: ${position_size * price:.2f})")
        
        # Create a market order
        logger.info(f"Creating market order: {symbol}, BUY, {position_size}")
        order_result = client.create_market_order(
            side="BUY",
            quantity=position_size
        )
        
        if order_result:
            logger.info(f"Test trade executed successfully: {order_result}")
            
            # Send trade notification
            telegram.send_trade_notification(
                trade_type='entry',
                symbol=symbol,
                side='BUY',
                quantity=position_size,
                price=price
            )
            
            # Wait a bit to let the order complete
            time.sleep(3)
            
            # Check open positions
            positions = client.get_open_positions()
            logger.info(f"Open positions after trade: {positions}")
            
            # Update account balance to see the effect of the trade
            new_balance = client.get_account_balance()
            logger.info(f"Updated balance after trade: {new_balance}")
            
            # Send account update notification
            telegram.send_account_update(
                balance=new_balance,
                open_positions=positions
            )
            
            # Close the position after a short delay
            time.sleep(5)
            closed = client.close_all_positions(symbol)
            logger.info(f"Position closing result: {closed}")
            
            # Send notification about closed position
            telegram.send_message_sync(f"✅ *TEST TRADE COMPLETED*\nTest position for {symbol} has been closed.")
            
            # Get final balance
            final_balance = client.get_account_balance()
            logger.info(f"Final balance after closing position: {final_balance}")
            
            return True
        else:
            logger.error("Failed to execute test trade")
            telegram.send_error("Test Trade", "Failed to execute test trade on testnet")
            return False
    
    except Exception as e:
        logger.error(f"Error executing test trade: {e}")
        telegram.send_error("Test Trade", f"Error: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("Starting testnet trade and Telegram notification test")
    
    # First test Telegram notifications
    telegram, notification_result = test_telegram_notification()
    
    if not notification_result:
        logger.error("Telegram notifications are not working. Please check your bot token and chat ID.")
        return
        
    logger.info("Telegram notifications are working! Proceeding with test trade...")
    
    # Execute test trade with Telegram notifications
    test_trade(telegram)
    
    logger.info("Test completed!")

if __name__ == "__main__":
    main()