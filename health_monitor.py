#!/usr/bin/env python3

import logging
import sys
import time
from datetime import datetime
import requests
import json
import websocket
import threading
from binance.client import Client
from telegram_notifier import TelegramNotifier
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("health_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HealthMonitor")

class HealthMonitor:
    """Monitor the health of the trading bot's components"""
    
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.ws_connections = {}
        self.api_client = None
        
    def check_api_connection(self):
        """Check if the Binance API is working"""
        try:
            if config.TEST_MODE:
                api_key = config.TEST_API_KEY
                api_secret = config.TEST_API_SECRET
                testnet = True
            else:
                api_key = config.REAL_API_KEY
                api_secret = config.REAL_API_SECRET
                testnet = False
                
            self.api_client = Client(api_key, api_secret, testnet=testnet)
            
            # Try to get server time
            time_data = self.api_client.get_server_time()
            
            # Verify we got valid data
            if time_data and 'serverTime' in time_data:
                server_time = datetime.fromtimestamp(time_data['serverTime']/1000)
                logger.info(f"API connection successful. Server time: {server_time}")
                return True
            else:
                logger.warning("API connection response format not as expected")
                return False
                
        except Exception as e:
            logger.error(f"API connection check failed: {e}")
            return False
    
    def check_websocket_connection(self, symbol="BTCUSDT"):
        """Check if WebSocket connection is working"""
        try:
            ws_connected = threading.Event()
            message_received = threading.Event()
            ws = None
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    logger.info(f"WebSocket message received: {message[:100]}...")
                    message_received.set()
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                ws_connected.set()
                
                # Subscribe to a stream
                stream_name = f"{symbol.lower()}@bookTicker"
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": int(time.time())
                }
                ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {stream_name}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed: {close_status_code}, {close_msg}")
            
            # Connect to WebSocket
            ws_url = "wss://fstream.binancefuture.com/ws" if not config.TEST_MODE else "wss://stream.binancefuture.com/ws"
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection in a thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and message
            ws_connected.wait(timeout=10)
            if not ws_connected.is_set():
                logger.error("Failed to connect to WebSocket")
                return False
                
            # Wait for a message to confirm connection is working
            message_received.wait(timeout=15)
            if not message_received.is_set():
                logger.error("No message received from WebSocket")
                return False
                
            # Close WebSocket
            ws.close()
            logger.info("WebSocket check completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket check failed: {e}")
            return False
    
    def run_health_check(self):
        """Run a complete health check and send notification"""
        logger.info("Running full health check...")
        
        # Check API connection
        api_status = self.check_api_connection()
        logger.info(f"API Status: {'OK' if api_status else 'ERROR'}")
        
        # Check WebSocket connection
        ws_status = self.check_websocket_connection()
        logger.info(f"WebSocket Status: {'OK' if ws_status else 'ERROR'}")
        
        # Send notification with status
        self.telegram.send_system_status_notification(api_status, ws_status)
        
        return api_status and ws_status
        
    def run_scheduled_checks(self, interval_hours=5):
        """Run health checks at scheduled intervals"""
        try:
            logger.info(f"Starting scheduled health checks every {interval_hours} hours")
            self.telegram.send_message(f"🔍 <b>Health Monitor Started</b>\nRunning checks every {interval_hours} hours")
            
            while True:
                # Run a health check
                self.run_health_check()
                
                # Sleep until next check
                logger.info(f"Next check in {interval_hours} hours")
                time.sleep(interval_hours * 3600)
                
        except KeyboardInterrupt:
            logger.info("Health monitor stopped by user")
        except Exception as e:
            logger.error(f"Error in health monitor: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Bot Health Monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Check interval in hours")
    parser.add_argument("--check-now", action="store_true", help="Run a check immediately")
    
    args = parser.parse_args()
    
    monitor = HealthMonitor()
    
    if args.check_now:
        # Run a single check
        result = monitor.run_health_check()
        sys.exit(0 if result else 1)
    else:
        # Run scheduled checks
        monitor.run_scheduled_checks(args.interval)
