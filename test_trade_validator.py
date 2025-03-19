import logging
import sys
from binance_client import BinanceClient
from risk_manager import RiskManager
from trade_validator import TradeValidator
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TradeValidatorTest")

def test_trade_validator():
    """Test trade validator functionality"""
    try:
        # Initialize components
        client = BinanceClient()
        risk_manager = RiskManager(
            client, 
            risk_percentage=1.0,  # 1% risk per trade
            stop_loss_percentage=2.0,  # 2% stop loss
            take_profit_percentage=3.0  # 3% take profit
        )
        validator = TradeValidator(client, risk_manager)
        
        # Test symbol
        symbol = "BTCUSDT"
        
        # Get current market price
        ticker = client.client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        logger.info(f"Current {symbol} price: {current_price}")
        
        # Test position size calculation
        position_size = validator.validate_and_calculate_position(symbol, current_price)
        if position_size:
            logger.info(f"Valid position size calculated: {position_size} {symbol}")
            
            # Test stop loss validation for long position
            sl_price = current_price * 0.98  # 2% below entry
            sl_valid = validator.validate_stop_loss(current_price, sl_price, "BUY")
            logger.info(f"Stop loss validation (LONG): Price={sl_price}, Valid={sl_valid}")
            
            # Test take profit validation for long position
            tp_price = current_price * 1.03  # 3% above entry
            tp_valid = validator.validate_take_profit(current_price, tp_price, "BUY")
            logger.info(f"Take profit validation (LONG): Price={tp_price}, Valid={tp_valid}")
            
            # Test stop loss validation for short position
            sl_price = current_price * 1.02  # 2% above entry
            sl_valid = validator.validate_stop_loss(current_price, sl_price, "SELL")
            logger.info(f"Stop loss validation (SHORT): Price={sl_price}, Valid={sl_valid}")
            
            # Test take profit validation for short position
            tp_price = current_price * 0.97  # 3% below entry
            tp_valid = validator.validate_take_profit(current_price, tp_price, "SELL")
            logger.info(f"Take profit validation (SHORT): Price={tp_price}, Valid={tp_valid}")
            
        else:
            logger.error("Failed to calculate valid position size")
            
        # Test with different risk levels
        test_risks = [0.5, 1.0, 2.0]  # Test 0.5%, 1%, and 2% risk
        for risk in test_risks:
            risk_manager.risk_percentage = risk
            position_size = validator.validate_and_calculate_position(symbol, current_price)
            if position_size:
                position_value = position_size * current_price
                logger.info(f"Risk {risk}%: Size={position_size} {symbol}, Value={position_value:.2f} USDT")
        
    except Exception as e:
        logger.error(f"Error testing trade validator: {e}")

if __name__ == "__main__":
    logger.info("Starting trade validator test...")
    test_trade_validator()
