import json
import websocket
import threading
import logging
import time
from datetime import datetime
import config

logger = logging.getLogger(__name__)

# Define WebSocket host
WS_HOST = "wss://fstream.binancefuture.com"

class BinanceWebsocket:
    """Binance Websocket client for receiving real-time market data"""
    
    def __init__(self, symbol, on_message_callback, on_close_callback=None):
        self.symbol = symbol.lower()
        self.on_message_callback = on_message_callback
        self.on_close_callback = on_close_callback
        self.ws = None
        self.thread = None
        self.is_connected = False
        self.subscriptions = []
        self.initialized = False
        
        # Initialize price tracking attributes
        self.latest_bid = 0
        self.latest_ask = 0
        self.latest_mark_price = 0
        
        # Use testnet or real endpoints depending on config
        if config.TEST_MODE:
            self.stream_url = f"{WS_HOST}/ws"  # Testnet WebSocket URL
            logger.info(f"Using Binance Futures Testnet WebSocket")
        else:
            self.stream_url = "wss://fstream.binance.com/ws"
            logger.info(f"Using Binance Futures Production WebSocket")
        
        # Add connection tracking
        self.connection_confirmed = threading.Event()
        self.connection_attempts = 0
        self.max_connection_attempts = 5
            
    def connect(self):
        """Connect to the websocket"""
        websocket.enableTrace(False)
        
        # Reset connection confirmation flag
        self.connection_confirmed.clear()
        self.connection_attempts += 1
        
        self.ws = websocket.WebSocketApp(
            self.stream_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
            on_ping=self._on_ping,
            on_pong=self._on_pong
        )
        
        self.thread = threading.Thread(target=self._run_websocket)
        self.thread.daemon = True
        self.thread.start()
        
        # Wait for connection to be established with longer timeout
        timeout = 15  # Increase timeout to 15 seconds
        if not self.connection_confirmed.wait(timeout=timeout):
            logger.error("Failed to confirm WebSocket connection within timeout")
            if self.connection_attempts < self.max_connection_attempts:
                logger.info(f"Retrying connection (attempt {self.connection_attempts + 1}/{self.max_connection_attempts})")
                return self.connect()  # Try again
            else:
                raise ConnectionError("Failed to connect to Binance websocket after maximum retries")
            
        logger.info(f"Connected to Binance websocket for {self.symbol}")
        return True
        
    def _run_websocket(self):
        """Run the websocket connection with error handling"""
        try:
            # Set ping interval to 20 seconds to keep connection alive
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Error in websocket thread: {e}")
            self.is_connected = False
            if self.on_close_callback:
                self.on_close_callback()
        
    def _on_message(self, ws, message):
        """Handle incoming websocket messages"""
        # First message received - mark as connected and initialized
        if not self.is_connected:
            self.is_connected = True
            self.initialized = True
            logger.info(f"First message received from WebSocket for {self.symbol}")
            
        try:
            data = json.loads(message)
            
            # Log received data with truncation for large payloads
            if 'stream' in data and 'data' in data:
                # This is a combined stream format
                stream = data['stream']
                stream_data = data['data']
                logger.debug(f"Received WebSocket data on stream {stream}: {json.dumps(stream_data)[:200]}...")
                
                # Extract event type if available
                event_type = stream_data.get('e', 'unknown')
                symbol = stream_data.get('s', 'unknown')
                
                # Log message summary based on event type
                if event_type == 'kline':
                    k = stream_data.get('k', {})
                    logger.info(f"Kline: {symbol} {k.get('i')} Open: {k.get('o')} Close: {k.get('c')} Volume: {k.get('v')}")
                elif event_type == 'bookTicker':
                    logger.info(f"BookTicker: {symbol} Bid: {stream_data.get('b')} Ask: {stream_data.get('a')}")
                elif event_type == 'markPriceUpdate':
                    logger.info(f"MarkPrice: {symbol} Price: {stream_data.get('p')} FundingRate: {stream_data.get('r', 'N/A')}")
                elif event_type == 'trade':
                    logger.info(f"Trade: {symbol} Price: {stream_data.get('p')} Quantity: {stream_data.get('q')}")
                else:
                    # For other event types, log a summary
                    logger.info(f"Received {event_type} event for {symbol}")
            else:
                # Regular message format
                if 'e' in data:
                    event_type = data['e']
                    symbol = data.get('s', 'unknown')
                    logger.info(f"Received {event_type} event for {symbol}")
                    
                    # Log specific details based on event type
                    if event_type == 'kline':
                        k = data.get('k', {})
                        logger.info(f"Kline: {symbol} {k.get('i')} Open: {k.get('o')} Close: {k.get('c')}")
                    elif event_type == 'bookTicker':
                        logger.info(f"BookTicker: {symbol} Bid: {data.get('b')} Ask: {data.get('a')}")
                    elif event_type == 'markPriceUpdate':
                        logger.info(f"MarkPrice: {symbol} Price: {data.get('p')}")
                    elif event_type == 'trade':
                        logger.info(f"Trade: {symbol} Price: {data.get('p')} Quantity: {data.get('q')}")
                elif 'b' in data and 'a' in data and 's' in data:
                    symbol = data.get('s', 'unknown')
                    logger.info(f"BookTicker: {symbol} Bid: {data.get('b')} Ask: {data.get('a')}")
                else:
                    # For messages without clear structure, log a truncated version
                    logger.info(f"Received message: {json.dumps(data)[:100]}...")
            
            # Handle book ticker updates
            if 'b' in data and 'a' in data and 's' in data:
                try:
                    # Only convert to float if we have string values
                    if isinstance(data['b'], str) and isinstance(data['a'], str):
                        self.latest_bid = float(data['b'])
                        self.latest_ask = float(data['a'])
                except ValueError:
                    pass
                
            # Handle mark price updates
            elif 'e' in data and data['e'] == 'markPriceUpdate' and 'p' in data:
                try:
                    if isinstance(data['p'], str):
                        self.latest_mark_price = float(data['p'])
                except ValueError:
                    pass
            
            # Pass the message to the callback
            if self.initialized:  # Only send messages after initialization is complete
                self.on_message_callback(data)
                
        except Exception as e:
            logger.error(f"Error processing websocket message: {e}")
            # Log the raw message that caused the error
            logger.error(f"Raw message: {message[:200]}...")
    
    def _on_error(self, ws, error):
        """Handle websocket errors"""
        logger.error(f"Websocket error for {self.symbol}: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle websocket connection close"""
        logger.info(f"Websocket connection closed for {self.symbol}: {close_status_code=}, {close_msg=}")
        self.is_connected = False
        
        if self.on_close_callback:
            self.on_close_callback()
            
    def _on_open(self, ws):
        """Handle websocket connection open"""
        logger.info(f"Websocket connection established for {self.symbol}")
        self.is_connected = True
        
        # Signal that connection is established
        self.connection_confirmed.set()
        
        # Add a small delay before marking as initialized
        def mark_initialized():
            self.initialized = True
            logger.info(f"WebSocket fully initialized for {self.symbol}")
            
        # Mark as initialized after a short delay
        threading.Timer(1.0, mark_initialized).start()
        
    def _on_ping(self, ws, message):
        """Handle ping from server"""
        logger.debug(f"Received ping from server for {self.symbol}")
        
    def _on_pong(self, ws, message):
        """Handle pong response"""
        logger.debug(f"Received pong from server for {self.symbol}")

    def subscribe_kline(self, symbol, interval):
        """Subscribe to kline/candlestick data"""
        try:
            symbol = symbol.lower().replace('/', '')
            stream_name = f"{symbol}@kline_{interval}"
            
            # Create subscription message
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            # Add to subscriptions list
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
            
            # Only send if connected
            if self.is_connected:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
                return True
            else:
                logger.warning(f"Cannot subscribe to {stream_name}: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to kline: {e}")
            return False
        
    def subscribe_trade(self, symbol):
        """Subscribe to trade data"""
        try:
            symbol = symbol.lower().replace('/', '')
            stream_name = f"{symbol}@trade"
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
                
            if self.is_connected:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
                return True
            else:
                logger.warning(f"Cannot subscribe to {stream_name}: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to trade: {e}")
            return False
        
    def subscribe_book_ticker(self, symbol):
        """Subscribe to book ticker data (best bid/ask)"""
        try:
            symbol = symbol.lower().replace('/', '')
            stream_name = f"{symbol}@bookTicker"
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
                
            if self.is_connected:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
                return True
            else:
                logger.warning(f"Cannot subscribe to {stream_name}: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to book ticker: {e}")
            return False
        
    def subscribe_mark_price(self, symbol):
        """Subscribe to mark price data"""
        try:
            symbol = symbol.lower().replace('/', '')
            stream_name = f"{symbol}@markPrice@1s"
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
                
            if self.is_connected:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
                return True
            else:
                logger.warning(f"Cannot subscribe to {stream_name}: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to mark price: {e}")
            return False
        
    def subscribe_depth(self, symbol, levels=5):
        """Subscribe to partial depth data"""
        try:
            symbol = symbol.lower().replace('/', '')
            stream_name = f"{symbol}@depth{levels}"
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
                
            if self.is_connected:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
                return True
            else:
                logger.warning(f"Cannot subscribe to {stream_name}: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to depth: {e}")
            return False
        
    def subscribe_all_market_data(self, symbol):
        """Subscribe to all relevant market data streams for a symbol"""
        success = True
        
        # Add delay between subscriptions to prevent rate limiting
        if self.subscribe_kline(symbol, '1m'):
            time.sleep(0.5)
        else:
            success = False
            
        if self.subscribe_book_ticker(symbol):
            time.sleep(0.5)
        else:
            success = False
            
        if self.subscribe_mark_price(symbol):
            time.sleep(0.5)
        else:
            success = False
        
        logger.info(f"Subscribed to all market data streams for {symbol}: {success}")
        return success
        
    def disconnect(self):
        """Disconnect from the websocket"""
        if self.ws:
            self.ws.close()
            logger.info("Websocket connection closed")
            
    def is_alive(self):
        """Check if the websocket connection is alive"""
        return self.thread and self.thread.is_alive() and self.is_connected
        
    def get_latest_price(self):
        """Get the latest price"""
        if self.latest_mark_price > 0:
            return self.latest_mark_price
        elif self.latest_ask > 0 and self.latest_bid > 0:
            return (self.latest_ask + self.latest_bid) / 2
        else:
            return 0