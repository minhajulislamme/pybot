#!/usr/bin/env python3

import json
import logging
import sys
import time
from datetime import datetime
import websocket
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebSocketChecker")

class WebSocketChecker:
    """Utility to check WebSocket connections and message structure"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol.lower()
        self.ws = None
        self.message_types = {}
        self.message_count = 0
        self.running = False
        
        # Use testnet or real endpoints depending on config
        if config.TEST_MODE:
            self.stream_url = "wss://fstream.binancefuture.com/ws"
            logger.info(f"Using Binance Futures Testnet WebSocket")
        else:
            self.stream_url = "wss://fstream.binance.com/ws"
            logger.info(f"Using Binance Futures Production WebSocket")

    def on_message(self, ws, message):
        """Process incoming messages and analyze their structure"""
        self.message_count += 1
        
        try:
            # Parse the message
            data = json.loads(message)
            
            # Determine message type
            msg_type = "unknown"
            if 'e' in data:
                msg_type = data['e']
            elif 'b' in data and 'a' in data and 's' in data:
                msg_type = "bookTicker"
                
            # Count message types
            if msg_type not in self.message_types:
                self.message_types[msg_type] = 1
            else:
                self.message_types[msg_type] += 1
                
            # Every 10 messages, print structure of different message types
            if self.message_count % 10 == 0:
                self.print_status()
                
                # Every 50 messages, print a detailed structure of one message
                if self.message_count % 50 == 0:
                    logger.info(f"Sample message structure for {msg_type}:\n{json.dumps(data, indent=2)}")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(f"Raw message: {message}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info(f"WebSocket connection closed: {close_status_code=}, {close_msg=}")
        self.running = False
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.running = True
        
        # Subscribe to multiple data streams
        self.subscribe_streams()
    
    def subscribe_streams(self):
        """Subscribe to various WebSocket streams for testing"""
        streams = [
            f"{self.symbol}@kline_1m",
            f"{self.symbol}@trade",
            f"{self.symbol}@bookTicker",
            f"{self.symbol}@markPrice@1s",
            f"{self.symbol}@depth5"
        ]
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to streams: {streams}")
    
    def print_status(self):
        """Print current WebSocket status and message statistics"""
        logger.info(f"Received {self.message_count} total messages")
        logger.info("Message types received:")
        for msg_type, count in self.message_types.items():
            logger.info(f"  - {msg_type}: {count}")
    
    def run(self, duration=60):
        """Run the WebSocket check for the specified duration"""
        logger.info(f"Starting WebSocket check for {self.symbol} for {duration} seconds")
        
        # Create and connect WebSocket
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.stream_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for WebSocket to connect
        timeout = 10
        start_time = time.time()
        while not self.running and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.running:
            logger.error("Failed to connect to WebSocket within timeout")
            return False
        
        # Run for the specified duration
        try:
            start_time = time.time()
            while time.time() - start_time < duration and self.running:
                time.sleep(1)
                
            # Print final status
            self.print_status()
            
            # Close WebSocket connection
            self.ws.close()
            logger.info("WebSocket check completed")
            return True
            
        except KeyboardInterrupt:
            logger.info("WebSocket check interrupted by user")
            self.ws.close()
            return False

if __name__ == "__main__":
    import threading
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Check Binance Futures WebSocket connection")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to monitor")
    parser.add_argument("--time", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()
    
    # Run the WebSocket checker
    checker = WebSocketChecker(args.symbol)
    checker.run(args.time)
