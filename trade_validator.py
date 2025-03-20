import logging
from decimal import Decimal, ROUND_DOWN
import config

logger = logging.getLogger(__name__)

class TradeValidator:
    """Validates trade parameters and calculates position sizes"""
    
    def __init__(self, client, risk_manager):
        self.client = client
        self.risk_manager = risk_manager
        
        # Cache for symbol info to reduce API calls
        self.symbol_info_cache = {}

    def get_symbol_leverage(self, symbol):
        """Get leverage for a specific symbol from config"""
        try:
            # Find leverage in trading pairs config
            for pair in config.TRADING_PAIRS:
                if pair.get("symbol") == symbol:
                    return pair.get("leverage", config.DEFAULT_LEVERAGE)
            return config.DEFAULT_LEVERAGE
        except Exception as e:
            logger.error(f"Error getting leverage for {symbol}: {e}")
            return config.DEFAULT_LEVERAGE
            
    def get_symbol_info(self, symbol):
        """Get symbol info with caching to reduce API calls"""
        if symbol not in self.symbol_info_cache:
            try:
                self.symbol_info_cache[symbol] = self.client.client.get_symbol_info(symbol)
            except Exception as e:
                logger.error(f"Error fetching symbol info for {symbol}: {e}")
                return None
                
        return self.symbol_info_cache[symbol]
        
    def get_quantity_precision(self, symbol):
        """Get the appropriate quantity precision for a symbol"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No symbol info found for {symbol}, using default precision of 3")
                return 3  # Default precision if info not available
                
            # Get lot size filter for quantity precision
            for filter_item in symbol_info['filters']:
                if filter_item['filterType'] == 'LOT_SIZE':
                    step_size = filter_item['stepSize']
                    # Convert step size to precision
                    precision = 0
                    if '.' in step_size:
                        decimal_part = step_size.split('.')[1].rstrip('0')
                        if decimal_part:
                            precision = len(decimal_part)
                    
                    logger.info(f"Determined quantity precision for {symbol}: {precision}")
                    return precision
            
            logger.warning(f"No LOT_SIZE filter found for {symbol}, using default precision of 3")
            return 3  # Default if no LOT_SIZE filter
            
        except Exception as e:
            logger.error(f"Error getting quantity precision for {symbol}: {e}")
            return 3  # Default in case of error
            
    def format_quantity(self, quantity, symbol):
        """Format quantity according to symbol precision requirements"""
        try:
            if symbol == 'BTCUSDT':
                # BTCUSDT requires exactly 3 decimal places
                qty = round(float(quantity), 3)
                return f"{qty:.3f}"
            elif symbol == 'ETHUSDT':
                # ETHUSDT requires exactly 3 decimal places
                qty = round(float(quantity), 3)
                return f"{qty:.3f}"
            elif symbol == 'SOLUSDT':
                # SOLUSDT requires whole numbers
                return str(int(float(quantity)))

            # For other symbols, get precision from exchange info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return f"{float(quantity):.3f}"

            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                return f"{float(quantity):.3f}"

            step_size = lot_size_filter['stepSize']
            precision = 0
            if '.' in step_size:
                decimal_part = step_size.split('.')[1].rstrip('0')
                precision = len(decimal_part) if decimal_part else 0

            qty = float(quantity)
            # Round to step size
            step_size = float(step_size)
            qty = round(qty / step_size) * step_size
            
            formatted_qty = f"{{:.{precision}f}}".format(qty)
            logger.info(f"Formatted {quantity} to {formatted_qty} with precision {precision} for {symbol}")
            return formatted_qty

        except Exception as e:
            logger.error(f"Error formatting quantity for {symbol}: {e}")
            return f"{float(quantity):.3f}"
            
    def validate_and_calculate_position(self, symbol, entry_price):
        """Calculate valid position size based on account balance and risk settings"""
        try:
            # Get account balance and convert to Decimal
            balance = Decimal(str(self.client.get_account_balance()))
            
            # Get leverage for this symbol
            leverage = Decimal(str(self.get_symbol_leverage(symbol)))
            
            # Get symbol information for precision
            symbol_info = self.get_symbol_info(symbol)
            
            if not symbol_info:
                raise ValueError(f"Could not get symbol info for {symbol}")
                
            # Get quantity precision from LOT_SIZE filter
            quantity_precision = self.get_quantity_precision(symbol)
            
            # Convert values to Decimal for precise calculation
            risk_percentage = Decimal(str(self.risk_manager.risk_percentage))
            entry_price = Decimal(str(entry_price))
            
            # Calculate position size based on risk percentage
            risk_amount = balance * (risk_percentage / Decimal('100'))
            max_position_size_usd = risk_amount * leverage
            
            # Set minimum notional values and quantity requirements per symbol
            min_notional = Decimal('25')  # Default minimum notional
            if symbol == 'BTCUSDT':
                min_notional = Decimal('100')  # Higher minimum for BTC
            elif symbol == 'ETHUSDT':
                min_notional = Decimal('25')
            elif symbol == 'SOLUSDT':
                min_notional = Decimal('25')
            
            # Calculate initial quantity
            quantity = max_position_size_usd / entry_price
            
            # Special handling for SOLUSDT
            if symbol == 'SOLUSDT':
                # Ensure minimum 1 unit for SOLUSDT and round up to whole number
                quantity = max(Decimal('1'), quantity.quantize(Decimal('1'), rounding=ROUND_DOWN))
            else:
                # Ensure we meet minimum notional value for other pairs
                if quantity * entry_price < min_notional:
                    quantity = (min_notional * Decimal('1.1')) / entry_price  # Add 10% buffer
                
                # Round down to respect quantity precision
                quantity = quantity.quantize(Decimal('0.' + '0' * quantity_precision), rounding=ROUND_DOWN)
            
            # Format quantity according to symbol requirements
            formatted_quantity = self.format_quantity(float(quantity), symbol)
            
            # Final validation of notional value
            final_quantity = Decimal(str(formatted_quantity))
            final_notional = final_quantity * entry_price
            
            # Ensure the final quantity is valid
            if final_quantity <= 0:
                logger.warning(f"Invalid quantity calculated: {final_quantity} {symbol}")
                return None
            
            if final_notional < min_notional:
                logger.warning(f"Position size too small. Minimum notional value: {min_notional} USDT")
                return None
            
            logger.info(f"Calculated position size: {formatted_quantity} {symbol} (Value: {float(final_notional):.2f} USDT) with {int(leverage)}x leverage")
            return float(formatted_quantity)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None

    def validate_stop_loss(self, entry_price, stop_loss_price, side):
        """Validate stop loss price"""
        if side == "BUY":
            if stop_loss_price >= entry_price:
                return False
            min_distance = entry_price * 0.001  # Minimum 0.1% distance
            return (entry_price - stop_loss_price) >= min_distance
        else:
            if stop_loss_price <= entry_price:
                return False
            min_distance = entry_price * 0.001
            return (stop_loss_price - entry_price) >= min_distance

    def validate_take_profit(self, entry_price, take_profit_price, side):
        """Validate take profit price"""
        if side == "BUY":
            if take_profit_price <= entry_price:
                return False
            min_distance = entry_price * 0.002  # Minimum 0.2% distance
            return (take_profit_price - entry_price) >= min_distance
        else:
            if take_profit_price >= entry_price:
                return False
            min_distance = entry_price * 0.002
            return (entry_price - take_profit_price) >= min_distance
