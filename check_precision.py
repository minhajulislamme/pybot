#!/usr/bin/env python3
import logging
from binance.client import Client
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_symbol_precision(symbol):
    """Check the precision requirements for a specific symbol"""
    try:
        # Initialize client
        if config.TEST_MODE:
            client = Client(config.TEST_API_KEY, config.TEST_API_SECRET, testnet=True)
        else:
            client = Client(config.REAL_API_KEY, config.REAL_API_SECRET)
        
        # Get symbol info
        symbol_info = client.get_symbol_info(symbol)
        
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return
        
        # Log all filters for reference
        logger.info(f"\nAll filters for {symbol}:")
        for filter_item in symbol_info['filters']:
            logger.info(f"{filter_item}")
        
        # Extract LOT_SIZE filter for quantity precision
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        
        if lot_size_filter:
            step_size = lot_size_filter['stepSize']
            # Calculate precision from step size
            if '.' in step_size:
                decimal_part = step_size.split('.')[1].rstrip('0')
                precision = len(decimal_part) if decimal_part else 0
            else:
                precision = 0
                
            min_qty = lot_size_filter['minQty']
            max_qty = lot_size_filter['maxQty']
            
            logger.info(f"\nPrecision details for {symbol}:")
            logger.info(f"Step Size: {step_size}")
            logger.info(f"Calculated Precision: {precision} decimal places")
            logger.info(f"Min Quantity: {min_qty}")
            logger.info(f"Max Quantity: {max_qty}")
            
            # Test formatting examples
            test_quantities = [0.123456789, 1.23456789, 12.3456789]
            logger.info("\nFormatting examples:")
            
            for qty in test_quantities:
                formatted_qty = format(float(qty), f".{precision}f")
                logger.info(f"Original: {qty} -> Formatted: {formatted_qty}")
        else:
            logger.error(f"LOT_SIZE filter not found for {symbol}")
    
    except Exception as e:
        logger.error(f"Error checking symbol precision: {e}")

if __name__ == "__main__":
    # Check precision for problem symbols
    symbols_to_check = ["SOLUSDT", "ETHUSDT", "ADAUSDT", "BTCUSDT"]
    
    for symbol in symbols_to_check:
        check_symbol_precision(symbol)
        print("\n" + "-" * 50 + "\n")  # Separator between symbols