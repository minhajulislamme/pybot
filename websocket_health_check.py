#!/usr/bin/env python3

import logging
import time
import sys
from datetime import datetime
from websocket_client import BinanceWebsocket
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebSocketHealthCheck")

class WebSocketHealthMonitor:
    """Utility to test and monitor WebSocket connections"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.message_counts = {
            'kline': 0,
            'trade': 0,
            'bookTicker': 0,
            'markPrice': 0,
            'depth': 0
        }
        self.last_message_time = None
        self.running = False

    def on_message(self, message):
        """Process incoming WebSocket messages"""
        try:
            # Update last message time
            self.last_message_time = datetime.now()
            
            # Identify message type and increment counter
            if 'e' in message:
                event_type = message['e']
                if event_type in self.message_counts:
                    self.message_counts[event_type] += 1
                elif event_type == 'kline':
                    self.message_counts['kline'] += 1
            elif 'b' in message and 'a' in message:
                self.message_counts['bookTicker'] += 1
                
            # Print progress every 50 messages
            total_messages = sum(self.message_counts.values())
            if total_messages % 50 == 0:
                self.print_status()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def on_close(self):
        """Handle WebSocket close"""
        logger.warning("WebSocket connection closed")
        if self.running:
            logger.info("Attempting to reconnect...")
            # Wait 5 seconds before reconnecting
            time.sleep(5)
            self.connect()

    def connect(self):
        """Connect to WebSocket and subscribe to data streams"""
        try:
            logger.info(f"Connecting to Binance WebSocket for {self.symbol}...")
            self.ws = BinanceWebsocket(self.symbol, self.on_message, self.on_close)
            self.ws.connect()
            
            # Subscribe to all data streams
            self.ws.subscribe_all_market_data(self.symbol)
            logger.info("Successfully connected and subscribed to all data streams")
            
            # Set running flag
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def print_status(self):
        """Print current status of message counts"""
        total = sum(self.message_counts.values())
        elapsed = (datetime.now() - self.start_time).total_seconds()
        msgs_per_sec = total / elapsed if elapsed > 0 else 0
        
        status = f"\n--- WebSocket Health Check for {self.symbol} ---\n"
        status += f"Running for: {elapsed:.1f} seconds\n"
        status += f"Total messages: {total} ({msgs_per_sec:.1f}/sec)\n"
        status += "Message counts:\n"
        
        for msg_type, count in self.message_counts.items():
            status += f"  - {msg_type}: {count}\n"
            
        status += f"Last message received: {self.last_message_time}\n"
        status += "Connection status: " + ("ALIVE" if self.ws.is_alive() else "DISCONNECTED")
        
        print(status)

    def run(self, duration=60):
        """Run the health check for the specified duration in seconds"""
        try:
            # Connect to WebSocket
            if not self.connect():
                return False
                
            # Record start time
            self.start_time = datetime.now()
            logger.info(f"Starting WebSocket health check for {duration} seconds...")
            
            # Run for the specified duration
            end_time = self.start_time + timedelta(seconds=duration)
            
            while datetime.now() < end_time and self.running:
                # Check if connection is still alive
                if not self.ws.is_alive():
                    logger.warning("WebSocket connection lost")
                    time.sleep(2)  # Give it a moment to reconnect
                    continue
                
                # Send ping every 20 seconds to keep connection alive
                if (datetime.now() - self.start_time).total_seconds() % 20 < 1:
                    self.ws.send_ping()
                
                time.sleep(1)
                
            # Print final status
            self.print_status()
            
            # Close WebSocket connection
            self.running = False
            self.ws.disconnect()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Health check interrupted by user")
            self.running = False
            if hasattr(self, 'ws'):
                self.ws.disconnect()
            return False
        except Exception as e:
            logger.error(f"Error during WebSocket health check: {e}")
            self.running = False
            if hasattr(self, 'ws'):
                self.ws.disconnect()
            return False

def main():
    """Run WebSocket health check"""
    from datetime import timedelta
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Binance Futures WebSocket Health Check")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol to monitor")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds to run the health check")
    args = parser.parse_args()
    
    logger.info(f"Starting WebSocket health check for {args.symbol} for {args.duration} seconds")
    
    # Run the health check
    monitor = WebSocketHealthMonitor(args.symbol)
    result = monitor.run(args.duration)
    
    if result:
        logger.info("WebSocket health check completed successfully")
        return 0
    else:
        logger.error("WebSocket health check failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
