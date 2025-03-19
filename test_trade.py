#!/usr/bin/env python3
import logging
import time
import sys
from binance_client import BinanceClient
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TestTrade")

def test_trade():
    """Execute a test trade cycle (buy, set stop loss, sell) on the Binance Futures testnet"""
    # Make sure TEST_MODE is enabled
    if not config.TEST_MODE:
        logger.error("Test mode is not enabled! Set TEST_MODE = True in config.py")
        return
    
    # Initialize the Binance client
    client = BinanceClient()
    
    # Choose symbol and trade parameters
    symbol = "BTCUSDT"  # You can change this to any other symbol
    
    try:
        # Get current price to calculate appropriate quantity
        ticker = client.client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Calculate quantity to meet minimum notional value of 100 USDT
        # Adding a buffer to ensure we're above the minimum
        min_notional = 100
        quantity = min_notional / current_price
        quantity = round(quantity, 3) + 0.001  # Round to 3 decimal places and add a small buffer
        
        logger.info(f"Current price of {symbol}: {current_price} USDT")
        logger.info(f"Trading quantity: {quantity} BTC (notional value: {quantity * current_price} USDT)")
        
        # Check account balance before trading
        balance_before = client.get_account_balance()
        logger.info(f"Account balance before trade: {balance_before} USDT")
        
        # Step 1: Execute a market buy order
        logger.info(f"Placing market BUY order for {quantity} {symbol}...")
        buy_order = client.place_market_order(symbol, "BUY", quantity)
        logger.info(f"BUY order executed: {buy_order}")
        
        # Wait for a moment to ensure the order is processed
        time.sleep(2)
        
        # Get position information using the correct method
        positions = client.get_all_open_positions()
        position = next((pos for pos in positions if pos['symbol'] == symbol.replace('/', '')), None)
        
        if not position:
            logger.error(f"Could not find open position for {symbol}")
            return
            
        entry_price = float(position['entryPrice'])
        logger.info(f"Position opened at price: {entry_price}")
        
        # Step 2: Set a stop loss (2% below entry price)
        stop_price = round(entry_price * 0.98, 2)
        logger.info(f"Setting stop loss at {stop_price}...")
        stop_loss_order = client.place_stop_loss(symbol, "BUY", quantity, stop_price)
        logger.info(f"Stop loss order placed: {stop_loss_order}")
        
        # Wait for 5 seconds before closing the position
        logger.info("Waiting for 5 seconds before closing the position...")
        time.sleep(5)
        
        # Step 3: Cancel all pending orders (including stop loss)
        logger.info("Cancelling all pending orders...")
        client.cancel_all_orders(symbol)
        
        # Step 4: Execute a market sell order to close the position
        logger.info(f"Placing market SELL order to close position...")
        sell_order = client.place_market_order(symbol, "SELL", quantity)
        logger.info(f"SELL order executed: {sell_order}")
        
        # Check account balance after trading
        time.sleep(2)
        balance_after = client.get_account_balance()
        logger.info(f"Account balance after trade: {balance_after} USDT")
        logger.info(f"Profit/Loss: {balance_after - balance_before} USDT")
        
        logger.info("Test trade completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test trade: {e}")
        
        # Try to close any open position in case of error
        try:
            logger.info("Attempting to close any open position...")
            positions = client.get_all_open_positions()
            position = next((pos for pos in positions if pos['symbol'] == symbol.replace('/', '')), None)
            
            if position:
                pos_quantity = abs(float(position['positionAmt']))
                
                if pos_quantity > 0:
                    logger.info(f"Found open position of {pos_quantity}. Closing...")
                    client.place_market_order(symbol, "SELL", pos_quantity)
                    logger.info(f"Position closed successfully")
        except Exception as close_error:
            logger.error(f"Error closing position: {close_error}")

if __name__ == "__main__":
    logger.info("Starting test trade on Binance Futures testnet...")
    test_trade()