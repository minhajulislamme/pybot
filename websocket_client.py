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
        
        # Initialize price tracking attributes
        self.latest_bid = 0
        self.latest_ask = 0
        self.latest_mark_price = 0
        
        # Use testnet or real endpoints depending on config
        if config.TEST_MODE:
            self.stream_url = f"{WS_HOST}/ws"  # Testnet WebSocket URL
            logger.info(f"Using Binance Futures Testnet WebSocket")
        else:
            self.stream_url = "wss://fstream.binancefuture.com/ws"
            logger.info(f"Using Binance Futures Production WebSocket")
            
    def connect(self):
        """Connect to the websocket"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.stream_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        self.thread = threading.Thread(target=self._run_websocket)
        self.thread.daemon = True
        self.thread.start()
        
        # Wait for connection to be established
        timeout = 10
        start_time = time.time()
        while not self.is_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.is_connected:
            logger.error("Failed to connect to Binance websocket within timeout")
            raise ConnectionError("Failed to connect to Binance websocket")
            
        logger.info("Connected to Binance websocket")
        
    def _run_websocket(self):
        """Run the websocket connection with error handling"""
        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"Error in websocket thread: {e}")
            if self.on_close_callback:
                self.on_close_callback()
        
    def _on_message(self, ws, message):
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            
            # Handle book ticker updates
            if 'b' in data and 'a' in data and 's' in data:
                self.latest_bid = float(data['b'])
                self.latest_ask = float(data['a'])
                
            # Handle mark price updates
            elif 'e' in data and data['e'] == 'markPriceUpdate':
                self.latest_mark_price = float(data['p'])
            
            self.on_message_callback(data)
        except Exception as e:
            logger.error(f"Error processing websocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle websocket errors"""
        logger.error(f"Websocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle websocket connection close"""
        logger.info(f"Websocket connection closed: {close_status_code=}, {close_msg=}")
        self.is_connected = False
        
        if self.on_close_callback:
            self.on_close_callback()
            
    def _on_open(self, ws):
        """Handle websocket connection open"""
        logger.info("Websocket connection established")
        self.is_connected = True
        
        # Subscribe to kline stream
        self.subscribe_kline(self.symbol, '1m')
        
    def subscribe_kline(self, symbol, interval):
        """Subscribe to kline/candlestick data"""
        symbol = symbol.lower().replace('/', '')
        stream_name = f"{symbol}@kline_{interval}"
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {stream_name}")
        
    def subscribe_trade(self, symbol):
        """Subscribe to trade data"""
        symbol = symbol.lower().replace('/', '')
        stream_name = f"{symbol}@trade"
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {stream_name}")
        
    def subscribe_book_ticker(self, symbol):
        """Subscribe to book ticker data (best bid/ask)"""
        symbol = symbol.lower().replace('/', '')
        stream_name = f"{symbol}@bookTicker"
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {stream_name}")
        
    def subscribe_mark_price(self, symbol):
        """Subscribe to mark price data"""
        symbol = symbol.lower().replace('/', '')
        stream_name = f"{symbol}@markPrice@1s"
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {stream_name}")
        
    def disconnect(self):
        """Disconnect from the websocket"""
        if self.ws:
            self.ws.close()
            logger.info("Websocket connection closed")
            
    def is_alive(self):
        """Check if the websocket connection is alive"""
        return self.thread and self.thread.is_alive() and self.is_connected