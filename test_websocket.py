#!/usr/bin/env python3

import logging
import sys
import time
import json
from datetime import datetime
import config
from websocket_client import BinanceWebsocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebSocketTest")

class WebSocketTester:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.message_count = 0
        self.start_time = None
        self.finished = False
        
    def on_message(self, message):
        """Process incoming websocket message"""
        self.message_count += 1
        
        # Print every 10th message
        if self.message_count % 10 == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Received {self.message_count} messages ({self.message_count/elapsed:.2f} msgs/sec)")
            
            # Print the actual message content occasionally
            if self.message_count % 50 == 0:
                if isinstance(message, dict):
                    # Determine message type
                    msg_type = "Unknown"
                    if 'e' in message:
                        msg_type = message['e']
                    elif 'k' in message:
                        msg_type = "Kline"
                    elif 'b' in message and 'a' in message and 's' in message:
                        msg_type = "BookTicker"
                        
                    logger.info(f"Sample {msg_type} message: {json.dumps(message, indent=2)}")
    
    def on_close(self):
        """Handle websocket connection close"""
        logger.warning("WebSocket connection closed")
        if not self.finished:
            logger.info("Attempting to reconnect...")
            self.test_websocket()  # Try to reconnect
    
    def test_websocket(self, duration=60):
        """Test WebSocket connection for the specified duration"""
        try:
            logger.info(f"Testing WebSocket connection for {self.symbol} for {duration} seconds")
            self.start_time = datetime.now()
            
            # Create and connect WebSocket
            ws = BinanceWebsocket(self.symbol, self.on_message, self.on_close)
            ws.connect()
            
            # Subscribe to streams
            logger.info("Subscribing to market data streams...")
            time.sleep(1)  # Give the connection time to stabilize
            
            ws.subscribe_kline(self.symbol, '1m')
            time.sleep(1)  # Add delay between subscriptions
            
            ws.subscribe_book_ticker(self.symbol)
            time.sleep(1)  # Add delay between subscriptions
            
            ws.subscribe_mark_price(self.symbol)
            logger.info("Subscribed to all test streams")
            
            # Run for the specified duration
            end_time = time.time() + duration
            while time.time() < end_time and ws.is_alive():
                # Print connection status every 10 seconds
                if int(time.time()) % 10 == 0:
                    logger.info(f"WebSocket is {'connected' if ws.is_alive() else 'disconnected'}")
                time.sleep(1)
                
            # Mark as finished to prevent reconnection attempts
            self.finished = True
            
            # Disconnect WebSocket
            ws.disconnect()
            
            # Print summary
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Test completed. Received {self.message_count} messages in {elapsed:.2f} seconds")
            logger.info(f"Average rate: {self.message_count/elapsed:.2f} messages/second")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            self.finished = True
            return False
        except Exception as e:
            logger.error(f"Error during WebSocket test: {e}")
            self.finished = True
            return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test Binance Futures WebSocket connection")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to test")
    parser.add_argument("--time", type=int, default=60, help="Test duration in seconds")
    args = parser.parse_args()
    
    # Run the test
    tester = WebSocketTester(args.symbol)
    tester.test_websocket(args.time)
