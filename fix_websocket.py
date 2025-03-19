#!/usr/bin/env python3

import logging
import sys
import time
import json
import threading
import websocket
import requests
from datetime import datetime
import config
from telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebSocketFixer")

class WebSocketFixer:
    """Utility to diagnose and fix WebSocket connection issues"""
    
    def __init__(self, notify=True):
        self.notify = notify
        if notify:
            self.telegram = TelegramNotifier()
    
    def check_basic_connectivity(self):
        """Check basic internet connectivity"""
        try:
            # Try to connect to Binance main site
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Basic connectivity to Binance REST API is working")
                return True
            else:
                logger.error(f"❌ Cannot connect to Binance REST API. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Network connectivity issue: {e}")
            return False
    
    def check_websocket_connectivity(self):
        """Check WebSocket connectivity"""
        connection_event = threading.Event()
        message_received = threading.Event()
        ws = None
        
        def on_message(ws, message):
            logger.info(f"✅ Received WebSocket message: {message[:100]}...")
            message_received.set()
            
        def on_error(ws, error):
            logger.error(f"❌ WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code}, {close_msg}")
            
        def on_open(ws):
            logger.info("✅ WebSocket connection established")
            connection_event.set()
            
            # Subscribe to a simple stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@bookTicker"],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info("Subscription request sent")
        
        try:
            # Use appropriate WebSocket endpoint based on config
            websocket_url = "wss://fstream.binancefuture.com/ws"
            if config.TEST_MODE:
                websocket_url = "wss://stream.binancefuture.com/ws"
            
            logger.info(f"Connecting to WebSocket: {websocket_url}")
            
            # Create WebSocket connection
            websocket.enableTrace(True)  # Enable tracing for detailed logs
            ws = websocket.WebSocketApp(
                websocket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection in a thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            connection_timeout = 15
            logger.info(f"Waiting up to {connection_timeout} seconds for WebSocket connection...")
            if connection_event.wait(timeout=connection_timeout):
                logger.info("WebSocket connected successfully")
                
                # Wait for message
                message_timeout = 15
                logger.info(f"Waiting up to {message_timeout} seconds for messages...")
                message_success = message_received.wait(timeout=message_timeout)
                
                # Close connection
                ws.close()
                
                if message_success:
                    logger.info("✅ WebSocket test PASSED - connection and messages working")
                    return True
                else:
                    logger.error("❌ WebSocket connected but no messages received")
                    return False
            else:
                logger.error("❌ Failed to establish WebSocket connection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in WebSocket test: {e}")
            return False
        finally:
            if ws:
                ws.close()
    
    def diagnose_and_fix(self):
        """Run diagnostics and recommend fixes"""
        logger.info("🔍 Starting WebSocket connection diagnostics")
        
        if self.notify:
            self.telegram.send_message("🔍 <b>WebSocket Diagnostic Started</b>\nChecking connection issues...")
        
        # Store diagnostic results
        results = {}
        
        # Check 1: Basic internet connectivity
        results['basic_connectivity'] = self.check_basic_connectivity()
        
        # Check 2: WebSocket connectivity
        results['websocket_connectivity'] = self.check_websocket_connectivity()
        
        # Analyze results and provide recommendations
        recommendations = []
        
        if not results['basic_connectivity']:
            recommendations.append("- Check your internet connection")
            recommendations.append("- Verify that you can access https://binance.com")
            recommendations.append("- Check if your network blocks secure connections")
        
        if not results['websocket_connectivity'] and results['basic_connectivity']:
            recommendations.append("- Your network might be blocking WebSocket connections")
            recommendations.append("- Try using a different network connection")
            recommendations.append("- Check if a firewall is blocking WebSocket traffic")
            recommendations.append("- Restart your network router/equipment")
        
        # Create diagnostic report
        report = "📊 <b>WebSocket Diagnostic Report</b>\n\n"
        report += f"Basic Connectivity: {'✅' if results['basic_connectivity'] else '❌'}\n"
        report += f"WebSocket Connectivity: {'✅' if results['websocket_connectivity'] else '❌'}\n\n"
        
        if recommendations:
            report += "<b>Recommendations:</b>\n"
            report += "\n".join(recommendations)
        else:
            report += "✅ No issues detected. If you're still experiencing problems, please restart the trading bot."
        
        logger.info(report.replace('<b>', '').replace('</b>', ''))
        
        if self.notify:
            self.telegram.send_message(report)
        
        return results['websocket_connectivity']
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix WebSocket Connection Issues")
    parser.add_argument("--no-notify", action="store_true", help="Don't send Telegram notifications")
    args = parser.parse_args()
    
    fixer = WebSocketFixer(notify=not args.no_notify)
    success = fixer.diagnose_and_fix()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
