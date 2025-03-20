import logging
import time
import hmac
import hashlib
from binance.client import Client
from binance.exceptions import BinanceAPIException
import config
import requests

logger = logging.getLogger(__name__)

# Define API hosts for testnet and production
TESTNET_API_HOST = "https://testnet.binancefuture.com"
PRODUCTION_API_HOST = "https://fapi.binance.com"

class BinanceClient:
    def __init__(self):
        self.test_mode = config.TEST_MODE
        self.notifications_enabled = getattr(config, 'NOTIFICATIONS_ENABLED', False)
        self.initialize_client()
        # Cache for symbol info to reduce API calls
        self.symbol_info_cache = {}

    def initialize_client(self):
        """Initialize the Binance client with appropriate API keys"""
        try:
            if self.test_mode:
                logger.info("Initializing Binance Futures client in TEST mode")
                self.api_key = config.TEST_API_KEY
                self.api_secret = config.TEST_API_SECRET
                self.client = Client(self.api_key, self.api_secret, testnet=True)
                self.api_url = TESTNET_API_HOST
            else:
                logger.info("Initializing Binance Futures client in REAL trading mode")
                self.api_key = config.REAL_API_KEY
                self.api_secret = config.REAL_API_SECRET
                self.client = Client(self.api_key, self.api_secret)
                self.api_url = PRODUCTION_API_HOST

            # Verify connection first
            self.client.futures_account_balance()
            logger.info("Successfully connected to Binance Futures API")

        except BinanceAPIException as e:
            logger.error(f"Error initializing Binance client: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Binance client: {e}")
            raise

    def set_leverage(self, symbol, leverage):
        """Set leverage for a specific symbol"""
        try:
            symbol = symbol.replace('/', '')
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=leverage
                    )
                    logger.info(f"Leverage set to {leverage}x for {symbol}")
                    return response
                except BinanceAPIException as e:
                    if e.code == -1000:  # Unknown error
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Retrying leverage setting for {symbol} (attempt {retry_count + 1})")
                            time.sleep(1)  # Add delay between retries
                            continue
                    logger.error(f"Error setting leverage for {symbol}: {e}")
                    return None  # Return None instead of raising to allow trading to continue
                except Exception as e:
                    logger.error(f"Unexpected error setting leverage for {symbol}: {e}")
                    return None
                
            logger.error(f"Failed to set leverage for {symbol} after {max_retries} attempts")
            return None
                    
        except Exception as e:
            logger.error(f"Error in set_leverage: {e}")
            return None  # Return None instead of raising to allow trading to continue
            
    def setup_trading_pairs(self):
        """Set up leverage for all trading pairs after successful client initialization"""
        success = True
        for pair_config in config.TRADING_PAIRS:
            symbol = pair_config.get("symbol")
            leverage = pair_config.get("leverage", config.DEFAULT_LEVERAGE)
            if symbol:
                result = self.set_leverage(symbol, leverage)
                if result is None:
                    logger.warning(f"Could not set leverage for {symbol}, will use exchange default")
                    success = False
                else:
                    logger.info(f"Successfully set {leverage}x leverage for {symbol}")
        return success

    def get_account_balance(self):
        """Get futures account balance"""
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'timestamp': timestamp
            }
            
            # Generate signature
            query_string = '&'.join([f"{key}={params[key]}" for key in params])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            # Make direct API request
            url = f"{self.api_url}/fapi/v2/balance"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                balances = response.json()
                # Find USDT balance
                usdt_balance = next((bal for bal in balances if bal['asset'] == 'USDT'), None)
                if usdt_balance:
                    return float(usdt_balance['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            raise

    def get_detailed_account_info(self):
        """Get detailed futures account information"""
        try:
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            
            # Generate signature
            query_string = '&'.join([f"{key}={params[key]}" for key in params])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            # Make direct API request
            url = f"{self.api_url}/fapi/v2/account"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting account info: {response.text}")
                return self.get_fallback_account_info()
        except Exception as e:
            logger.error(f"Error getting detailed account info: {e}")
            return self.get_fallback_account_info()

    def get_fallback_account_info(self):
        """Provide fallback account information when API fails"""
        try:
            # Get balance using native Binance API
            balance = self.get_account_balance()
            
            # Create a simplified account info structure with essential fields
            account_info = {
                'totalWalletBalance': str(balance),
                'totalUnrealizedProfit': '0',
                'availableBalance': str(balance),
                'assets': []
            }
            
            logger.info("Using fallback account information")
            return account_info
        except Exception as e:
            logger.error(f"Error getting fallback account info: {e}")
            # Return minimal structure to prevent further errors
            return {
                'totalWalletBalance': '1000',  # Default value
                'totalUnrealizedProfit': '0',
                'availableBalance': '1000',
                'assets': []
            }

    def get_all_open_positions(self):
        """Get all open positions"""
        try:
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            
            # Generate signature
            query_string = '&'.join([f"{key}={params[key]}" for key in params])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            # Make direct API request with the correct endpoint path
            endpoint = "/fapi/v2/positionRisk" if not self.test_mode else "/fapi/v1/positionRisk"
            url = f"{self.api_url}{endpoint}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                positions = response.json()
                # Filter out positions with zero amount
                return [pos for pos in positions if float(pos['positionAmt']) != 0]
            return []
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            raise

    def get_position_info(self, symbol):
        """Get current position information for a symbol"""
        try:
            symbol = symbol.replace('/', '')
            positions = self.get_all_open_positions()
            position = next((item for item in positions if item['symbol'] == symbol), None)
            return position
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            raise

    def calculate_position_size(self, price, symbol=None):
        """Calculate position size based on risk percentage"""
        try:
            balance = self.get_account_balance()
            risk_amount = balance * (config.RISK_PERCENTAGE / 100)
            
            # Calculate position size based on leverage
            symbol = symbol or config.TRADING_PAIRS[0]["symbol"]
            
            # Find the leverage for this symbol
            leverage = None
            for pair in config.TRADING_PAIRS:
                if pair.get("symbol") == symbol:
                    leverage = pair.get("leverage", config.DEFAULT_LEVERAGE)
                    break
            
            if leverage is None:
                leverage = config.DEFAULT_LEVERAGE
            
            position_size = (risk_amount * leverage) / price
            
            # Special handling for SOLUSDT
            if symbol == 'SOLUSDT':
                # Ensure minimum 1 unit and round to whole number
                position_size = max(1, int(position_size))
                return position_size
            
            # For other symbols, use standard precision handling
            symbol_info = self.client.get_symbol_info(symbol)
            step_size = 0.001  # Default if not found
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    break
                    
            precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
            position_size = round(position_size, precision)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise

    def get_symbol_info(self, symbol):
        """Get symbol info with caching to reduce API calls"""
        if symbol not in self.symbol_info_cache:
            try:
                self.symbol_info_cache[symbol] = self.client.get_symbol_info(symbol)
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
            # Handle specific symbols with known precision requirements
            if symbol == 'BTCUSDT':
                return f"{float(quantity):.3f}"  # Exactly 3 decimal places
            elif symbol == 'ETHUSDT':
                return f"{float(quantity):.3f}"  # Exactly 3 decimal places
            elif symbol == 'SOLUSDT':
                return str(int(float(quantity)))  # Convert to whole number
            
            # For other symbols, get precision from exchange info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No symbol info found for {symbol}, using default precision of 3")
                return f"{float(quantity):.3f}"
            
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                logger.warning(f"No LOT_SIZE filter found for {symbol}, using default precision of 3")
                return f"{float(quantity):.3f}"
            
            step_size = lot_size_filter['stepSize']
            precision = 0
            
            # Extract precision from step size
            if '.' in step_size:
                decimal_part = step_size.split('.')[1].rstrip('0')
                precision = len(decimal_part) if decimal_part else 0
            
            # Round to step size
            step_size = float(step_size)
            qty = float(quantity)
            qty = round(qty / step_size) * step_size
            
            # Format with proper precision
            formatted_qty = f"{{:.{precision}f}}".format(qty)
            logger.info(f"Formatted {quantity} to {formatted_qty} with precision {precision} for {symbol}")
            return formatted_qty
            
        except Exception as e:
            logger.error(f"Error formatting quantity for {symbol}: {e}")
            return f"{float(quantity):.3f}"  # Fallback with reasonable precision

    def place_market_order(self, symbol, side, quantity):
        """Place a market order"""
        try:
            symbol = symbol.replace('/', '')
            
            # Add extra validation for SOLUSDT
            if symbol == 'SOLUSDT':
                quantity = max(1, int(float(quantity)))  # Ensure minimum 1 unit and whole number
            
            # Format quantity with proper precision
            formatted_quantity = self.format_quantity(quantity, symbol)
            
            # Final validation to prevent zero quantity
            if float(formatted_quantity) <= 0:
                raise ValueError(f"Invalid quantity: {formatted_quantity} for {symbol}")
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=formatted_quantity
            )
            logger.info(f"Market order placed: {side} {formatted_quantity} {symbol}")
            
            # Send notification
            if self.notifications_enabled and config.NOTIFICATION_TYPES.get('trade_entry', False):
                self.send_telegram_notification(
                    f"🔄 New {side} Order\n"
                    f"Symbol: {symbol}\n"
                    f"Quantity: {formatted_quantity}\n"
                    f"Type: Market\n"
                    f"Mode: {'TEST' if self.test_mode else 'REAL'}"
                )
            
            return order
            
        except Exception as e:
            error_msg = f"Error placing market order: {e}"
            logger.error(error_msg)
            
            # Send error notification
            if self.notifications_enabled and config.NOTIFICATION_TYPES.get('error', False):
                self.send_telegram_notification(
                    f"❌ Error placing market order\n"
                    f"Symbol: {symbol}\n"
                    f"Error: {str(e)}"
                )
            
            raise

    def place_limit_order(self, symbol, side, quantity, price):
        """Place a limit order"""
        try:
            symbol = symbol.replace('/', '')
            
            # Format quantity with proper precision
            formatted_quantity = self.format_quantity(quantity, symbol)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=formatted_quantity,
                price=str(price)
            )
            logger.info(f"Limit order placed: {side} {formatted_quantity} {symbol} @ {price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing limit order: {e}")
            raise

    def place_stop_loss(self, symbol, side, quantity, stop_price, price=None):
        """Place a stop loss order"""
        try:
            symbol = symbol.replace('/', '')
            
            # Format quantity with proper precision
            formatted_quantity = self.format_quantity(quantity, symbol)
            
            params = {
                'symbol': symbol,
                'side': 'SELL' if side == 'BUY' else 'BUY',  # Opposite of position side
                'type': 'STOP_MARKET',
                'stopPrice': str(stop_price),
                'closePosition': 'true'
            }
            
            order = self.client.futures_create_order(**params)
            logger.info(f"Stop loss placed at {stop_price} for {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing stop loss: {e}")
            raise

    def place_take_profit(self, symbol, side, quantity, stop_price, price=None):
        """Place a take profit order"""
        try:
            symbol = symbol.replace('/', '')
            
            # Format quantity with proper precision
            formatted_quantity = self.format_quantity(quantity, symbol)
            
            params = {
                'symbol': symbol,
                'side': 'SELL' if side == 'BUY' else 'BUY',  # Opposite of position side
                'type': 'TAKE_PROFIT_MARKET',
                'stopPrice': str(stop_price),
                'closePosition': 'true'
            }
            
            order = self.client.futures_create_order(**params)
            logger.info(f"Take profit placed at {stop_price} for {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing take profit: {e}")
            raise

    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol"""
        try:
            symbol = symbol.replace('/', '')
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"Cancelled all orders for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"Error cancelling orders: {e}")
            raise

    def get_historical_klines(self, symbol, interval, start_time, end_time=None, limit=1000):
        """Get historical candlestick data"""
        try:
            symbol = symbol.replace('/', '')
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
            return klines
        except BinanceAPIException as e:
            logger.error(f"Error getting historical klines: {e}")
            raise

    def send_telegram_notification(self, message):
        """Send notification via Telegram"""
        if not self.notifications_enabled:
            return
            
        if not hasattr(config, 'TELEGRAM_BOT_TOKEN') or not hasattr(config, 'TELEGRAM_CHAT_ID'):
            logger.warning("Telegram credentials not configured. Skipping notification.")
            return
            
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            if not response.ok:
                logger.error(f"Failed to send Telegram notification: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")