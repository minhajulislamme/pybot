#!/usr/bin/env python3
import logging
from binance.client import Client
import config
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_exchange_info():
    """Get exchange info and extract precision details for specific symbols"""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    
    try:
        # Initialize client - using testnet since that's where the errors are occurring
        client = Client(config.TEST_API_KEY, config.TEST_API_SECRET, testnet=True)
        
        # Get exchange info - this contains all the precision settings from Binance
        exchange_info = client.futures_exchange_info()
        
        # Extract symbol info for our target symbols
        for sym_info in exchange_info['symbols']:
            if sym_info['symbol'] in symbols:
                logger.info(f"\n===== {sym_info['symbol']} =====")
                logger.info(f"Base Asset: {sym_info.get('baseAsset', 'N/A')}")
                logger.info(f"Quote Asset: {sym_info.get('quoteAsset', 'N/A')}")
                logger.info(f"Status: {sym_info.get('status', 'N/A')}")
                
                # Get the precision info - safely with get() to handle missing keys
                logger.info(f"Base Asset Precision: {sym_info.get('baseAssetPrecision', 'N/A')}")
                logger.info(f"Quote Precision: {sym_info.get('quoteAssetPrecision', sym_info.get('quotePrecision', 'N/A'))}")
                logger.info(f"Price Precision: {sym_info.get('pricePrecision', 'N/A')}")
                logger.info(f"Quantity Precision: {sym_info.get('quantityPrecision', 'N/A')}")
                
                # Get filters
                logger.info("\nFilters:")
                for filter_item in sym_info.get('filters', []):
                    if filter_item['filterType'] in ['LOT_SIZE', 'MARKET_LOT_SIZE', 'MIN_NOTIONAL']:
                        logger.info(json.dumps(filter_item, indent=2))
                        
        # Let's also directly check a trade with a test order but don't execute
        logger.info("\n\nDirect Quantity Check:")
        for symbol in symbols:
            # Test with different precision values
            test_values = []
            if symbol == 'BTCUSDT':
                test_values = [('0.001', '3 decimal places'), ('0.01', '2 decimal places')]
            elif symbol == 'ETHUSDT':
                test_values = [('0.01', '2 decimal places'), ('0.1', '1 decimal place')]
            elif symbol == 'SOLUSDT':
                test_values = [('1', '0 decimal places'), ('0.1', '1 decimal place')]
            elif symbol == 'ADAUSDT':
                test_values = [('10', '0 decimal places'), ('1', '0 decimal places')]
                
            for qty, desc in test_values:
                logger.info(f"Testing {symbol} with quantity {qty} ({desc})")
                # Just check the current price to see if we get any errors
                ticker = client.futures_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                logger.info(f"Current price for {symbol}: {price}")
                
    except Exception as e:
        logger.error(f"Error in script: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Checking direct precision information from Binance Futures API...")
    check_exchange_info()