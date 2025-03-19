#!/usr/bin/env python3

import logging
import time
import sys
import json
import threading
import websocket
from datetime import datetime
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebSocketMonitor")

class WebSocketMonitor:
    """Monitor WebSocket connections for multiple symbols using a single connection"""
    
    def __init__(self, symbols=None):
        if symbols is None:
            # Get symbols from config
            self.symbols = [pair['symbol'].lower() for pair in config.TRADING_PAIRS]
        else:
            self.symbols = [s.lower() for s in symbols]
            
        self.ws = None
        self.is_connected = False
        self.message_count = 0
        self.last_message_time = None
        self.message_types = {}
        self.symbol_counts = {symbol: 0 for symbol in self.symbols}
        
        # Use testnet or real endpoints depending on config
        if config.TEST_MODE:
            self.stream_url = "wss://stream.binancefuture.com/stream"
            logger.info(f"Using Binance Futures Testnet WebSocket")
        else:
            self.stream_url = "wss://fstream.binance.com/stream"
            logger.info(f"Using Binance Futures Production WebSocket")
    
    def on_message(self, ws, message):
        """Process incoming WebSocket messages"""
        self.message_count += 1
        self.last_message_time = datetime.now()
        
        try:
            # Parse the message
            data = json.loads(message)
            
            # Check if it's a stream data message
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                stream_data = data['data']
                
                # Extract symbol from stream name
                parts = stream.split('@')
                if len(parts) > 0:
                    symbol = parts[0]
                    if symbol in self.symbol_counts:
                        self.symbol_counts[symbol] += 1
                
                # Determine message type
                msg_type = "unknown"
                if 'e' in stream_data:
                    msg_type = stream_data['e']
                elif 'kline' in stream:
                    msg_type = "kline"
                elif 'bookTicker' in stream:
                    msg_type = "bookTicker"
                
                # Count message types
                if msg_type not in self.message_types:
                    self.message_types[msg_type] = 1
                else:
                    self.message_types[msg_type] += 1
                    
            # Print status periodically
            if self.message_count % 50 == 0:
                self.print_status()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.warning(f"WebSocket connection closed: {close_status_code=}, {close_msg=}")
        self.is_connected = False
        
        # Try to reconnect if unexpected close
        if close_status_code != 1000:  # Normal closure
            logger.info("Attempting to reconnect...")
            time.sleep(5)
            self.connect()
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        
        # Subscribe to streams for all symbols
        self.subscribe_streams()
    
    def subscribe_streams(self):
        """Subscribe to data streams for all symbols"""
        # List to hold all stream names
        streams = []
        
        # Add streams for each symbol
        for symbol in self.symbols:
            streams.extend([
                f"{symbol}@kline_1m",
                f"{symbol}@bookTicker",
                f"{symbol}@markPrice@1s"
            ])
        
        # Create subscription message
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        # Send the subscription request
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(streams)} streams for {len(self.symbols)} symbols")
    
    def print_status(self):
        """Print WebSocket connection status"""
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        status = f"\n=== WebSocket Monitor Status ===\n"
        status += f"Running for: {elapsed:.1f} seconds\n"
        status += f"Total messages: {self.message_count}\n"
        status += f"Messages/sec: {self.message_count / elapsed:.2f}\n"
        status += f"Last message: {self.last_message_time}\n"
        status += f"Connected: {self.is_connected}\n\n"
        
        status += "Message Types:\n"
        for msg_type, count in sorted(self.message_types.items(), key=lambda x: x[1], reverse=True):
            status += f"  {msg_type}: {count}\n"
            
        status += "\nSymbol Message Counts:\n"
        for symbol, count in sorted(self.symbol_counts.items(), key=lambda x: x[1], reverse=True):
            status += f"  {symbol}: {count}\n"
            
        logger.info(status)
    
    def connect(self):
        """Connect to the WebSocket and start monitoring"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.stream_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
            'ping_interval': 30,
            'ping_timeout': 10
        })
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.is_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.is_connected:
            logger.error("Failed to connect to WebSocket within timeout")
            return False
            
        return True
    
    def run(self, duration=300):
        """Run the monitor for a specified duration"""
        logger.info(f"Starting WebSocket monitor for {len(self.symbols)} symbols")
        self.start_time = datetime.now()
        
        if not self.connect():
            return False
        
        try:
            # Run for the specified duration
            end_time = time.time() + duration
            while time.time() < end_time and self.is_connected:
                time.sleep(1)
                
            # Print final status
            self.print_status()
            
            # Close WebSocket
            self.ws.close()
            logger.info("WebSocket monitor completed")
            return True
            
        except KeyboardInterrupt:
            logger.info("WebSocket monitor interrupted by user")
            self.ws.close()
            return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Monitor Binance Futures WebSocket connections")
    parser.add_argument("--time", type=int, default=300, help="Duration in seconds (default: 300)")
    args = parser.parse_args()
    
    # Run the monitor
    monitor = WebSocketMonitor()
    monitor.run(args.time)
