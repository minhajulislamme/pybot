#!/usr/bin/env python3

import json
import logging
import sys
import time
import signal
from datetime import datetime
import argparse
from websocket_client import BinanceWebsocket
import config

# Configure logging to show more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WebSocketViewer")

class WebSocketDataViewer:
    """Utility to display real-time WebSocket data in a readable format"""
    
    def __init__(self, symbol, show_raw=False):
        self.symbol = symbol.upper()
        self.show_raw = show_raw
        self.running = True
        self.message_counts = {
            'kline': 0,
            'bookTicker': 0,
            'markPriceUpdate': 0,
            'trade': 0,
            'other': 0
        }
        self.start_time = datetime.now()
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n\nShutdown signal received, closing WebSocket...")
        self.running = False
        if hasattr(self, 'ws'):
            self.ws.disconnect()
        self.print_summary()
        sys.exit(0)
        
    def on_message(self, message):
        """Process and display incoming websocket messages"""
        try:
            # Handle combined stream format
            if isinstance(message, dict) and 'stream' in message and 'data' in message:
                stream = message['stream']
                data = message['data']
                logger.info(f"Stream: {stream}")
                self.display_message(data)
            else:
                # Handle regular format
                self.display_message(message)
                
            # Print summary every 100 messages
            total_messages = sum(self.message_counts.values())
            if total_messages > 0 and total_messages % 100 == 0:
                self.print_summary()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def display_message(self, data):
        """Display message data in a readable format"""
        try:
            # Determine message type
            if 'e' in data:
                event_type = data['e']
                self.message_counts[event_type] = self.message_counts.get(event_type, 0) + 1
            elif 'b' in data and 'a' in data and 's' in data:
                event_type = 'bookTicker'
                self.message_counts['bookTicker'] += 1
            else:
                event_type = 'other'
                self.message_counts['other'] += 1
            
            # Display raw message if requested
            if self.show_raw:
                logger.info(f"Raw message: {json.dumps(data)}")
                return
            
            # Format based on event type
            if event_type == 'kline':
                k = data.get('k', {})
                symbol = k.get('s', 'N/A')
                interval = k.get('i', 'N/A')
                is_closed = k.get('x', False)
                open_price = k.get('o', 'N/A')
                high_price = k.get('h', 'N/A')
                low_price = k.get('l', 'N/A')
                close_price = k.get('c', 'N/A')
                volume = k.get('v', 'N/A')
                
                candle_info = f"{'CLOSED ' if is_closed else ''}CANDLE: {symbol} {interval} │ "
                candle_info += f"O: {open_price} │ H: {high_price} │ L: {low_price} │ C: {close_price} │ Vol: {volume}"
                logger.info(candle_info)
                
            elif event_type == 'bookTicker':
                symbol = data.get('s', 'N/A')
                bid_price = data.get('b', 'N/A')
                bid_qty = data.get('B', 'N/A')
                ask_price = data.get('a', 'N/A')
                ask_qty = data.get('A', 'N/A')
                
                book_info = f"BOOK: {symbol} │ Bid: {bid_price} ({bid_qty}) │ Ask: {ask_price} ({ask_qty})"
                logger.info(book_info)
                
            elif event_type == 'markPriceUpdate':
                symbol = data.get('s', 'N/A')
                mark_price = data.get('p', 'N/A')
                index_price = data.get('i', 'N/A')
                funding_rate = data.get('r', 'N/A')
                
                mark_info = f"MARK: {symbol} │ Price: {mark_price} │ Index: {index_price} │ Funding: {funding_rate}"
                logger.info(mark_info)
                
            elif event_type == 'trade':
                symbol = data.get('s', 'N/A')
                price = data.get('p', 'N/A')
                quantity = data.get('q', 'N/A')
                buyer = data.get('b', 'N/A')
                seller = data.get('a', 'N/A')
                is_buyer_maker = data.get('m', False)
                direction = "SELL" if is_buyer_maker else "BUY"
                
                trade_info = f"TRADE: {symbol} │ {direction} │ Price: {price} │ Qty: {quantity}"
                logger.info(trade_info)
                
            else:
                # For other event types, just show type and symbol if available
                symbol = data.get('s', 'N/A')
                logger.info(f"EVENT: {event_type} │ {symbol}")
                
        except Exception as e:
            logger.error(f"Error displaying message: {e}")
            
    def print_summary(self):
        """Print summary of received messages"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        total = sum(self.message_counts.values())
        
        summary = "\n===== WebSocket Message Summary =====\n"
        summary += f"Symbol: {self.symbol}\n"
        summary += f"Running for: {elapsed:.1f} seconds\n"
        summary += f"Total messages: {total}\n"
        summary += f"Messages per second: {total / elapsed:.2f}\n"
        summary += "Message types:\n"
        
        for msg_type, count in self.message_counts.items():
            if count > 0:
                summary += f"  - {msg_type}: {count} ({count / total * 100:.1f}%)\n"
                
        logger.info(summary)
        
    def connect(self):
        """Connect to WebSocket and subscribe to data streams"""
        logger.info(f"Connecting to Binance Futures WebSocket for {self.symbol}...")
        self.ws = BinanceWebsocket(self.symbol, self.on_message)
        self.ws.connect()
        
        # Subscribe to data streams
        time.sleep(1)  # Allow connection to stabilize
        
        logger.info("Subscribing to data streams...")
        self.ws.subscribe_kline(self.symbol, '1m')
        time.sleep(0.5)
        
        self.ws.subscribe_book_ticker(self.symbol)
        time.sleep(0.5)
        
        self.ws.subscribe_mark_price(self.symbol)
        time.sleep(0.5)
        
        self.ws.subscribe_trade(self.symbol)
        logger.info(f"Subscribed to all data streams for {self.symbol}")
        
    def run(self):
        """Run the WebSocket data viewer"""
        try:
            self.connect()
            
            logger.info(f"\nStarted WebSocket data viewer for {self.symbol}")
            logger.info("Press Ctrl+C to exit")
            
            # Keep the main thread running
            while self.running and self.ws.is_alive():
                time.sleep(1)
                
            return True
            
        except KeyboardInterrupt:
            logger.info("\nViewer stopped by user")
            if hasattr(self, 'ws'):
                self.ws.disconnect()
            self.print_summary()
            return False
        except Exception as e:
            logger.error(f"Error in WebSocket data viewer: {e}")
            if hasattr(self, 'ws'):
                self.ws.disconnect()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display Binance Futures WebSocket data in real-time")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--raw", action="store_true", help="Show raw JSON messages")
    
    args = parser.parse_args()
    
    viewer = WebSocketDataViewer(args.symbol, args.raw)
    viewer.run()
