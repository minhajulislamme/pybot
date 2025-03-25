"""
WebSocket client for real-time Binance data.
"""

import json
import logging
import threading
import websocket
import time
import queue
from collections import deque
from datetime import datetime, timedelta
from config.config import config

logger = logging.getLogger(__name__)

class BinanceWebsocketClient:
    def __init__(self, symbols=None, callbacks=None, buffer_size=1000):
        """Initialize with message buffering"""
        self.symbols = symbols or config.symbols
        self.timeframe = "1h"  # Default timeframe
        self.callbacks = callbacks or {}
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.ws_url = config.ws_url
        self.reconnect_delay = 1
        self.max_reconnect_delay = 300
        self.message_buffer = {}
        self.buffer_size = buffer_size
        self.message_queue = queue.Queue()
        self.processor_thread = None
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30
        self.streams = []  # Initialize streams list
        
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
    
    def _on_message(self, ws, message):
        """Queue message for processing"""
        try:
            self.message_queue.put(message)
        except Exception as e:
            logger.error(f"Error queueing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors with reconnection logic"""
        logger.error(f"WebSocket error: {error}")
        
        # Check if we need to reconnect
        if self.running:
            # Use exponential backoff for reconnection
            delay = min(self.reconnect_delay, self.max_reconnect_delay)
            logger.info(f"Attempting to reconnect in {delay} seconds...")
            time.sleep(delay)
            self.reconnect_delay *= 2  # Double the delay for next attempt
            self._connect()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure with reconnection"""
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
        
        # Attempt to reconnect if closed unexpectedly
        if self.running:
            delay = min(self.reconnect_delay, self.max_reconnect_delay)
            logger.info(f"Attempting to reconnect in {delay} seconds...")
            time.sleep(delay)
            self.reconnect_delay *= 2
            self._connect()
    
    def _on_open(self, ws):
        """Initialize connection and subscribe to streams"""
        logger.info("WebSocket connection established")
        self.reconnect_delay = 1  # Reset reconnection delay
        self.last_heartbeat = time.time()
        
        # Subscribe to streams
        if self.streams:
            subscribe_msg = json.dumps({
                "method": "SUBSCRIBE",
                "params": self.streams,
                "id": 1
            })
            ws.send(subscribe_msg)
            logger.info(f"Subscribed to streams: {', '.join(self.streams)}")
    
    def _heartbeat(self):
        """Send periodic heartbeat to keep connection alive"""
        while self.running:
            try:
                if time.time() - self.last_heartbeat > self.heartbeat_interval:
                    if self.ws and self.ws.sock:
                        self.ws.send("ping")
                        self.last_heartbeat = time.time()
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
        """Connect to WebSocket and setup streams"""
        try:
            # Create streams for each symbol
            streams = []
            for symbol in self.symbols:
                symbol = symbol.lower()
                streams.extend([
                    f"{symbol}@kline_{self.timeframe}",  # Kline/candlestick stream
                    f"{symbol}@bookTicker",  # Best bid/ask stream
                    f"{symbol}@aggTrade"     # Aggregated trades stream
                ])
            
            # Store streams for later use in reconnection/subscription    
            self.streams = streams
                
            # Create WebSocket URL - using stream path instead of combined URL
            # This is more reliable for multiple streams
            stream_url = f"{self.ws_url}/stream"
            
            # Setup WebSocket connection
            self.ws = websocket.WebSocketApp(
                stream_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            # Use ping/pong for better keep-alive
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            time.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            if self.running:
                self._connect()  # Attempt to reconnect
    
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