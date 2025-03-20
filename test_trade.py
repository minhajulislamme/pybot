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

def get_min_qty_for_notional(symbol, min_notional=25):
    """Calculate minimum quantity needed for min_notional value"""
    client = BinanceClient()
    ticker = client.client.futures_symbol_ticker(symbol=symbol)
    current_price = float(ticker['price'])
    
    # Use correct precision values based on Binance API
    if symbol == 'BTCUSDT':
        # BTCUSDT requires minimum notional of 100 USDT + 20% buffer for price movements
        min_qty = 120 / current_price  # Using 120 to ensure we stay above 100 USDT with price fluctuations
        # Round up to next 0.001 increment to ensure minimum notional
        qty_steps = int(min_qty * 1000)  # Convert to steps of 0.001
        if min_qty * 1000 > qty_steps:  # If there's any remainder, take next step
            qty_steps += 1
        qty_rounded = qty_steps / 1000  # Convert back to BTC quantity
        qty_str = f"{qty_rounded:.3f}"  # Format with exactly 3 decimal places
        logger.info(f"Using exact notional calculation for BTCUSDT: {qty_rounded * current_price:.2f} USDT")
    elif symbol == 'ETHUSDT':
        # ETHUSDT requires 3 decimal places, minimum 0.001
        qty = (min_notional / current_price) * 1.1  # Add 10% buffer
        qty_rounded = max(0.001, float('%.3f' % qty))  # Use string formatting to enforce precision
        qty_str = f"{qty_rounded:.3f}"
    elif symbol == 'BNBUSDT':
        # BNBUSDT requires 2 decimal places
        qty = (min_notional / current_price) * 1.1  # Add 10% buffer
        qty_rounded = max(0.01, float('%.2f' % qty))  # Use string formatting to enforce precision
        qty_str = f"{qty_rounded:.2f}"
        logger.info(f"Using 2 decimal precision for BNBUSDT: {qty_rounded * current_price:.2f} USDT")
    elif symbol == 'SOLUSDT':
        # SOLUSDT requires whole numbers only
        qty = (min_notional / current_price) * 1.1  # Add 10% buffer
        qty_rounded = max(1, int(qty))  # Round up to nearest whole number
        qty_str = str(qty_rounded)
        logger.info(f"Using whole number quantity for SOLUSDT: {qty_rounded} (value: {qty_rounded * current_price:.2f} USDT)")
    else:
        # Default handling with API precision check
        info = client.client.futures_exchange_info()
        symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            qty_precision = next((int(f['stepSize'].find('1') - 1) for f in symbol_info['filters'] 
                               if f['filterType'] == 'LOT_SIZE' and f['stepSize'].find('1') > 0), 3)
        else:
            qty_precision = 3  # Default precision if not found
            
        qty = (min_notional / current_price) * 1.1  # Add 10% buffer
        qty_rounded = float(f'%.{qty_precision}f' % qty)  # Use dynamic precision
        qty_str = f'%.{qty_precision}f' % qty_rounded
    
    logger.info(f"Min quantity for {symbol}: {qty_str} (at price: {current_price})")
    return qty_str, current_price

def test_trade(symbol):
    """Execute a test trade cycle (buy, set stop loss, sell) for a specific symbol"""
    # Make sure TEST_MODE is enabled
    if not config.TEST_MODE:
        logger.error("Test mode is not enabled! Set TEST_MODE = True in config.py")
        return False
    
    logger.info(f"======= Starting test for {symbol} =======")
    
    # Initialize the Binance client
    client = BinanceClient()
    
    try:
        # Get symbol info for price filters
        symbol_info = client.client.futures_exchange_info()
        symbol_rules = next((s for s in symbol_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_rules:
            logger.error(f"Could not find trading rules for {symbol}")
            return False
            
        # Get price filters
        percent_filter = next((f for f in symbol_rules['filters'] if f['filterType'] == 'PERCENT_PRICE'), None)
        if percent_filter:
            logger.info(f"Price filters for {symbol}: {percent_filter}")
        
        # Symbol-specific minimum notional values based on Binance requirements
        min_notional = 100 if symbol == 'BTCUSDT' else 25
        
        # Calculate minimum quantity needed based on current price
        quantity_str, current_price = get_min_qty_for_notional(symbol, min_notional=min_notional)
        
        # Convert to float for calculations but keep string for order placement
        quantity = float(quantity_str)
        
        notional_value = quantity * current_price
        logger.info(f"Current price of {symbol}: {current_price} USDT")
        logger.info(f"Trading quantity: {quantity_str} (notional value: {notional_value:.4f} USDT)")
        
        # Check account balance before trading
        balance_before = client.get_account_balance()
        logger.info(f"Account balance before trade: {balance_before} USDT")
        
        # For other symbols, execute normal market order
        logger.info(f"Placing market BUY order for {quantity_str} {symbol}...")
        buy_order = client.client.futures_create_order(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quantity=quantity_str
        )
        
        logger.info(f"BUY order executed: {buy_order}")
        
        # Wait longer to ensure the order is processed
        logger.info("Waiting for position to be registered...")
        time.sleep(8)  # Longer wait time
        
        # Get position information with retry mechanism
        position = None
        retry_count = 0
        max_retries = 4
        
        while retry_count < max_retries and not position:
            positions = client.get_all_open_positions()
            position = next((pos for pos in positions if pos['symbol'] == symbol), None)
            
            if not position:
                logger.warning(f"No position found yet for {symbol}, retrying... ({retry_count+1}/{max_retries})")
                retry_count += 1
                time.sleep(5)  # Longer wait between retries
                
        if not position:
            logger.error(f"Could not find open position for {symbol} after {max_retries} attempts")
            logger.info(f"Raw positions data: {positions}")
            # Try to get position details directly from order
            logger.info("Checking order status...")
            try:
                order_id = buy_order.get('orderId') 
                if order_id:
                    order_status = client.client.futures_get_order(symbol=symbol, orderId=order_id)
                    logger.info(f"Order status: {order_status}")
                    # Check if order was filled
                    if order_status.get('status') == 'FILLED':
                        logger.info(f"Order was filled with quantity {order_status.get('executedQty')}. Testing close position directly.")
                        # Try to close position directly
                        sell_order = client.client.futures_create_order(
                            symbol=symbol,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity_str
                        )
                        logger.info(f"SELL order executed: {sell_order}")
                        return True
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
            return False
            
        entry_price = float(position['entryPrice'])
        position_amt = float(position['positionAmt'])
        logger.info(f"Position opened at price: {entry_price}")
        logger.info(f"Position amount: {position_amt}")
        
        # Step 2: Set a stop loss (2% below entry price)
        stop_price = round(entry_price * 0.98, 2)
        logger.info(f"Setting stop loss at {stop_price}...")
        stop_loss_order = client.place_stop_loss(symbol, "BUY", abs(position_amt), stop_price)
        logger.info(f"Stop loss order placed: {stop_loss_order}")
        
        # Wait before closing the position
        logger.info("Waiting for 3 seconds before closing the position...")
        time.sleep(3)
        
        # Step 3: Cancel all pending orders (including stop loss)
        logger.info("Cancelling all pending orders...")
        client.cancel_all_orders(symbol)
        
        # Step 4: Execute a market sell order to close the position
        logger.info(f"Placing market SELL order to close position...")
        sell_order = client.place_market_order(symbol, "SELL", abs(position_amt))
        logger.info(f"SELL order executed: {sell_order}")
        
        # Check account balance after trading
        time.sleep(3)
        balance_after = client.get_account_balance()
        logger.info(f"Account balance after trade: {balance_after} USDT")
        logger.info(f"Profit/Loss: {balance_after - balance_before:.6f} USDT")
        
        logger.info(f"Test trade for {symbol} completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test trade for {symbol}: {e}")
        
        # Try to close any open position in case of error
        try:
            logger.info("Attempting to close any open position...")
            positions = client.get_all_open_positions()
            position = next((pos for pos in positions if pos['symbol'] == symbol), None)
            
            if position:
                pos_quantity = abs(float(position['positionAmt']))
                
                if pos_quantity > 0:
                    logger.info(f"Found open position of {pos_quantity}. Closing...")
                    client.place_market_order(symbol, "SELL", pos_quantity)
                    logger.info(f"Position closed successfully")
        except Exception as close_error:
            logger.error(f"Error closing position: {close_error}")
        
        return False

if __name__ == "__main__":
    logger.info("Starting test trades on Binance Futures testnet...")
    
    # List of symbols to test
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    
    # Test each symbol
    results = {}
    for symbol in symbols:
        success = test_trade(symbol)
        results[symbol] = "SUCCESS" if success else "FAILED"
        time.sleep(2)  # Brief pause between tests
    
    # Print summary
    logger.info("\n============= TEST SUMMARY =============")
    for symbol, result in results.items():
        logger.info(f"{symbol}: {result}")
    logger.info("=======================================")