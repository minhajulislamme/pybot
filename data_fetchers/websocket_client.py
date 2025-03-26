"""
WebSocket client for real-time Binance data with enhanced error handling and reconnection logic.
"""
import json
import logging
import threading
import websocket
import time
import queue
import ssl
from collections import deque
from datetime import datetime, timedelta
from config.config import config

# Configure module loggers
logger = logging.getLogger(__name__)
logging.getLogger('websocket').setLevel(logging.INFO)
websocket.enableTrace(False)  # Disable websocket debug tracing globally

class BinanceWebsocketClient:
    def __init__(self, symbols=None, callbacks=None, buffer_size=1000):
        """Initialize with message buffering and enhanced connection handling"""
        self.symbols = symbols or config.symbols
        self.timeframe = "1h"  # Default timeframe
        self.callbacks = callbacks or {}
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.ws_url = config.ws_url  # Use WebSocket URL from config
        self.reconnect_delay = 1
        self.max_reconnect_delay = 300
        self.message_buffer = {}
        self.buffer_size = buffer_size
        self.message_queue = queue.Queue()
        self.processor_thread = None
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30
        self.streams = []
        self.connection_retries = 0
        self.max_connection_retries = 10
        self.last_stream_activity = time.time()
        self.activity_timeout = 60  # Consider connection dead if no activity for 60 seconds
        self.use_combined_streams = True  # Flag to switch between stream methods
        
        # Initialize buffers for different stream types
        for symbol in self.symbols:
            self.message_buffer[f"{symbol.lower()}_trade"] = deque(maxlen=buffer_size)
            self.message_buffer[f"{symbol.lower()}_kline"] = deque(maxlen=buffer_size)
            self.message_buffer[f"{symbol.lower()}_book"] = deque(maxlen=buffer_size)

    def is_connected(self):
        """Check if websocket connection is active
        
        Returns:
            bool: True if websocket is connected, False otherwise
        """
        return self.running and self.ws is not None and hasattr(self.ws, 'sock') and self.ws.sock is not None
    
    def _process_messages(self):
        """Process messages from queue in background"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._handle_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_message(self, message_data):
        """Process different types of messages"""
        try:
            data = json.loads(message_data)
            
            if 'e' not in data:
                return
                
            event_type = data['e']
            symbol = data['s'].lower() if 's' in data else None
            
            # Store in appropriate buffer
            if event_type == 'aggTrade' and symbol:
                buffer_key = f"{symbol}_trade"
                self.message_buffer[buffer_key].append({
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'time': data['T'],
                    'is_buyer_maker': data['m']
                })
                
            elif event_type == 'kline' and symbol:
                buffer_key = f"{symbol}_kline"
                kline = data['k']
                self.message_buffer[buffer_key].append({
                    'open_time': kline['t'],
                    'close_time': kline['T'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']
                })
                
            elif event_type == 'bookTicker' and symbol:
                buffer_key = f"{symbol}_book"
                self.message_buffer[buffer_key].append({
                    'bid_price': float(data['b']),
                    'bid_qty': float(data['B']),
                    'ask_price': float(data['a']),
                    'ask_qty': float(data['A']),
                    'time': int(time.time() * 1000)
                })
            
            # Call registered callbacks
            if event_type in self.callbacks:
                self.callbacks[event_type](data)
                
        except json.JSONDecodeError:
            logger.error("Failed to decode message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors with improved reconnection logic"""
        logger.error(f"WebSocket error: {error}")
        
        if isinstance(error, websocket.WebSocketConnectionClosedException):
            logger.info("Connection was closed, attempting to reconnect...")
            self.reconnect_delay = 1  # Reset delay on connection close
        
        # Check if we need to reconnect
        if self.running and self.connection_retries < self.max_connection_retries:
            # If we keep getting errors and not using combined streams yet, try that approach
            if self.connection_retries >= 2 and not self.use_combined_streams:
                logger.info("Switching to combined streams URL after errors")
                self.use_combined_streams = True
                
            delay = min(self.reconnect_delay, self.max_reconnect_delay)
            logger.info(f"Attempting to reconnect in {delay} seconds... (Attempt {self.connection_retries+1}/{self.max_connection_retries})")
            time.sleep(delay)
            self.reconnect_delay *= 2  # Double the delay for next attempt
            self._connect()
        elif self.connection_retries >= self.max_connection_retries:
            logger.error("Max connection retries reached. Stopping WebSocket client.")
            self.stop()

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure with improved reconnection logic"""
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
        
        # Clear the current WebSocket instance
        self.ws = None
        
        # Attempt to reconnect if closed unexpectedly and within retry limits
        if self.running and self.connection_retries < self.max_connection_retries:
            delay = min(self.reconnect_delay, self.max_reconnect_delay)
            logger.info(f"Attempting to reconnect in {delay} seconds...")
            time.sleep(delay)
            self.reconnect_delay *= 2
            self._connect()
        elif self.connection_retries >= self.max_connection_retries:
            logger.error("Max connection retries reached. Stopping WebSocket client.")
            self.stop()

    def _on_open(self, ws):
        """Initialize connection and subscribe to streams"""
        logger.info("WebSocket connection established")
        self.reconnect_delay = 1  # Reset reconnection delay
        self.connection_retries = 0  # Reset retry counter on successful connection
        self.last_heartbeat = time.time()
        self.last_stream_activity = time.time()
        
        # Only send subscription if not using combined streams URL
        if not self.use_combined_streams:
            # Subscribe to streams
            subscribe_request = {
                "method": "SUBSCRIBE",
                "params": self.streams,
                "id": int(time.time() * 1000)
            }
            ws.send(json.dumps(subscribe_request))
            logger.info(f"Sent subscription request for streams: {', '.join(self.streams)}")

    def _heartbeat(self):
        """Send periodic heartbeat and monitor connection health"""
        while self.running:
            try:
                current_time = time.time()
                
                # Send ping if needed
                if current_time - self.last_heartbeat > self.heartbeat_interval:
                    if self.ws and self.ws.sock:
                        self.ws.send("ping")
                        self.last_heartbeat = current_time
                        logger.debug("Sent ping to keep connection alive")
                
                # Check for activity timeout
                if current_time - self.last_stream_activity > self.activity_timeout:
                    logger.warning(f"No WebSocket activity for {self.activity_timeout} seconds, forcing reconnection")
                    if self.ws:
                        self.ws.close()
                        
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
    
    def get_trade_data(self, symbol, lookback_seconds=None):
        """Get recent trade data from buffer"""
        buffer_key = f"{symbol.lower()}_trade"
        if buffer_key not in self.message_buffer:
            return []
            
        trades = list(self.message_buffer[buffer_key])
        if lookback_seconds:
            cutoff_time = int(time.time() * 1000) - (lookback_seconds * 1000)
            trades = [t for t in trades if t['time'] >= cutoff_time]
        return trades
    
    def get_kline_data(self, symbol, lookback_seconds=None):
        """Get recent kline data from buffer"""
        buffer_key = f"{symbol.lower()}_kline"
        if buffer_key not in self.message_buffer:
            return []
            
        klines = list(self.message_buffer[buffer_key])
        if lookback_seconds:
            cutoff_time = int(time.time() * 1000) - (lookback_seconds * 1000)
            klines = [k for k in klines if k['open_time'] >= cutoff_time]
        return klines
    
    def get_orderbook_data(self, symbol):
        """Get latest orderbook data"""
        buffer_key = f"{symbol.lower()}_book"
        if buffer_key not in self.message_buffer or not self.message_buffer[buffer_key]:
            return None
        return self.message_buffer[buffer_key][-1]
    
    def _connect(self):
        """Connect to WebSocket and setup streams with enhanced error handling"""
        try:
            # Reset connection retries if we've been connected before
            if self.ws:
                self.connection_retries = 0
                
            # Create streams for each symbol
            streams = []
            for symbol in self.symbols:
                symbol = symbol.lower()
                streams.extend([
                    f"{symbol}@kline_{self.timeframe}",
                    f"{symbol}@bookTicker",
                    f"{symbol}@aggTrade"
                ])
            
            # Store streams for later use
            self.streams = streams
            
            # Log streams being used
            logger.info(f"Setting up connection for streams: {streams}")
            
            # Choose between combined streams URL or subscription method
            if self.use_combined_streams:
                # Combined streams approach
                combined_stream = '/'.join(streams)
                stream_url = f"{self.ws_url}/stream?streams={combined_stream}"
                logger.info(f"Using combined stream URL method: {stream_url}")
            else:
                # Multiple streams using subscription method
                stream_url = f"{self.ws_url}/stream"
                logger.info(f"Using subscription method with endpoint: {stream_url}")
            
            # Setup WebSocket connection with appropriate options
            self.ws = websocket.WebSocketApp(
                stream_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            # Run with more aggressive ping/pong settings to detect disconnections earlier
            self.ws.run_forever(
                ping_interval=15,
                ping_timeout=10,
                ping_payload='{"ping": true}',
                sslopt={"cert_reqs": ssl.CERT_NONE}
            )
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.connection_retries += 1
            
            if self.connection_retries < self.max_connection_retries:
                delay = min(self.reconnect_delay, self.max_reconnect_delay)
                logger.info(f"Attempting to reconnect in {delay} seconds... (Attempt {self.connection_retries}/{self.max_connection_retries})")
                time.sleep(delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
                # If we've tried the subscription method several times, switch to combined streams
                if self.connection_retries >= 3 and not self.use_combined_streams:
                    logger.info("Switching to combined streams URL method after repeated connection failures")
                    self.use_combined_streams = True
                
                if self.running:
                    self._connect()
            else:
                logger.error("Max connection retries reached. Stopping WebSocket client.")
                self.stop()

    def _run_websocket(self):
        """Run WebSocket in a loop with automatic reconnection"""
        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket error in run loop: {e}")
        finally:
            if self.running:
                logger.info("Attempting to reconnect WebSocket...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                self._connect()
    
    def start(self):
        """Start WebSocket client with message processor"""
        if self.running:
            logger.warning("WebSocket client already running")
            return
            
        logger.info("Starting WebSocket client")
        self.running = True
        
        # Start message processor thread
        self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processor_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.heartbeat_thread.start()
        
        # Create WebSocket connection in a new thread
        # We'll use a new thread here instead of _run_websocket to avoid double initialization
        self.ws_thread = threading.Thread(target=self._connect, daemon=True)
        self.ws_thread.start()
        
        # Give time for the connection to establish
        time.sleep(0.5)
        
        logger.info("WebSocket client started")
    
    def stop(self):
        """Stop WebSocket client and clean up"""
        if not self.running:
            logger.warning("WebSocket client not running")
            return
            
        logger.info("Stopping WebSocket client")
        self.running = False
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
            
        # Wait for threads to finish
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
            
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
            
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
            
        self.ws = None
        self.ws_thread = None
        self.processor_thread = None
        self.heartbeat_thread = None
        
        # Clear buffers
        for key in self.message_buffer:
            self.message_buffer[key].clear()
            
        logger.info("WebSocket client stopped")
    
    def _on_ping(self, ws, message):
        """Handle ping messages from the server"""
        self.last_heartbeat = time.time()
        if message and hasattr(ws, "sock") and ws.sock:
            ws.sock.pong(message)
    
    def _on_pong(self, ws, message):
        """Handle pong responses"""
        self.last_heartbeat = time.time()
        logger.debug("Received pong from server")
    
    def _on_message(self, ws, message):
        """Queue message for processing with improved handling"""
        try:
            # Update activity timestamp
            self.last_stream_activity = time.time()
            
            data = json.loads(message)
            
            # Handle subscription response or other non-data messages
            if 'result' in data or 'id' in data:
                if 'result' in data and data['result'] is None:
                    logger.info("Successfully subscribed to streams")
                return
                
            # Handle combined streams format
            if 'stream' in data:
                # Queue the actual market data from the stream
                self.message_queue.put(json.dumps(data['data']))
            else:
                # Direct market data or other message types
                self.message_queue.put(message)

        except json.JSONDecodeError:
            logger.error(f"Failed to decode message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error in message handler: {e}")