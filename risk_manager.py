import logging
import config

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk Management class to handle position sizing and risk control"""
    
    def __init__(self, client, risk_percentage=None, stop_loss_percentage=None, take_profit_percentage=None):
        self.client = client
        # Use provided parameters or default from config
        self.risk_percentage = risk_percentage if risk_percentage is not None else config.DEFAULT_RISK_PERCENTAGE
        self.stop_loss_percentage = stop_loss_percentage if stop_loss_percentage is not None else config.DEFAULT_STOP_LOSS_PERCENTAGE
        self.take_profit_percentage = take_profit_percentage if take_profit_percentage is not None else config.DEFAULT_TAKE_PROFIT_PERCENTAGE
        logger.info(f"Risk Manager initialized: risk={self.risk_percentage}%, SL={self.stop_loss_percentage}%, TP={self.take_profit_percentage}%")
    
    def set_leverage_safely(self, symbol, leverage):
        """Safely set leverage with fallback options"""
        try:
            # First try to set the specified leverage
            response = self.client.set_leverage(symbol, leverage)
            if response:
                logger.info(f"[{symbol}] Successfully set leverage to {leverage}x")
                return True
                
            # If that fails with None response, try default leverage
            if leverage != config.DEFAULT_LEVERAGE:
                logger.warning(f"[{symbol}] Failed to set custom leverage {leverage}x, trying default {config.DEFAULT_LEVERAGE}x")
                response = self.client.set_leverage(symbol, config.DEFAULT_LEVERAGE)
                if response:
                    logger.info(f"[{symbol}] Successfully set default leverage {config.DEFAULT_LEVERAGE}x")
                    return True
            
            logger.warning(f"[{symbol}] Could not set leverage, will use exchange default")
            return False
            
        except Exception as e:
            logger.error(f"[{symbol}] Error setting leverage: {e}")
            return False

    def calculate_position_size(self, entry_price, symbol=None):
        """Calculate the position size based on account balance and risk parameters"""
        try:
            # Get account balance
            balance = self.client.get_account_balance()
            logger.info(f"Account Balance: {balance} USDT")
            
            # Calculate the risk amount in USDT
            risk_amount = balance * (self.risk_percentage / 100)
            logger.info(f"Risk amount per trade: {risk_amount:.2f} USDT")
            
            # Use the symbol-specific leverage if provided, otherwise use default
            symbol = symbol or config.TRADING_PAIRS[0]["symbol"]  # Default to first trading pair if not specified
            
            # Find the leverage for this symbol
            leverage = None
            for pair in config.TRADING_PAIRS:
                if pair.get("symbol") == symbol:
                    leverage = pair.get("leverage", config.DEFAULT_LEVERAGE)
                    break
            
            if leverage is None:
                leverage = config.DEFAULT_LEVERAGE
            
            # Calculate position size based on entry price and leverage
            position_size_in_usd = risk_amount * leverage
            
            # Convert to quantity based on entry price
            quantity = position_size_in_usd / entry_price
            
            # Round to appropriate precision for the asset
            symbol_info = self.client.client.get_symbol_info(symbol)
            step_size = 0.001  # Default if not found
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    break
                    
            precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
            position_size = round(quantity, precision)
            
            logger.info(f"Calculated position size: {position_size} {symbol} (≈{position_size_in_usd:.2f} USDT)")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise
    
    def calculate_stop_loss_price(self, entry_price, position_side):
        """Calculate stop loss price based on entry price and position side"""
        if position_side.upper() == 'BUY':
            # For long positions, stop loss is below entry price
            stop_loss_price = entry_price * (1 - self.stop_loss_percentage / 100)
        else:
            # For short positions, stop loss is above entry price
            stop_loss_price = entry_price * (1 + self.stop_loss_percentage / 100)
        
        logger.info(f"Stop Loss price calculated: {stop_loss_price:.2f} for {position_side} position at entry {entry_price:.2f}")
        return stop_loss_price
    
    def calculate_take_profit_price(self, entry_price, position_side):
        """Calculate take profit price based on entry price and position side"""
        if position_side.upper() == 'BUY':
            # For long positions, take profit is above entry price
            take_profit_price = entry_price * (1 + self.take_profit_percentage / 100)
        else:
            # For short positions, take profit is below entry price
            take_profit_price = entry_price * (1 - self.take_profit_percentage / 100)
        
        logger.info(f"Take Profit price calculated: {take_profit_price:.2f} for {position_side} position at entry {entry_price:.2f}")
        return take_profit_price
    
    def set_stop_loss_and_take_profit(self, entry_price, position_side, quantity, symbol=None):
        """Set stop loss and take profit orders"""
        try:
            symbol = symbol or config.TRADING_PAIRS[0]["symbol"]  # Default to first trading pair if not specified
            side = position_side.upper()
            
            # Calculate stop loss and take profit prices
            sl_price = self.calculate_stop_loss_price(entry_price, side)
            tp_price = self.calculate_take_profit_price(entry_price, side)
            
            # Place stop loss order
            sl_order = self.client.place_stop_loss(symbol, side, quantity, sl_price)
            logger.info(f"Stop loss order placed for {symbol}: {sl_order}")
            
            # Place take profit order
            tp_order = self.client.place_take_profit(symbol, side, quantity, tp_price)
            logger.info(f"Take profit order placed for {symbol}: {tp_order}")
            
            return {
                'stop_loss_order': sl_order,
                'take_profit_order': tp_order
            }
            
        except Exception as e:
            logger.error(f"Error setting stop loss and take profit: {e}")
            raise
    
    def check_max_drawdown(self, initial_balance, current_balance, max_drawdown_percentage=25):
        """Check if the current drawdown exceeds the maximum allowed drawdown"""
        if initial_balance <= 0:
            return True
            
        drawdown_percentage = ((initial_balance - current_balance) / initial_balance) * 100
        
        if drawdown_percentage >= max_drawdown_percentage:
            logger.warning(f"Maximum drawdown reached: {drawdown_percentage:.2f}% > {max_drawdown_percentage}%")
            return True
        
        return False