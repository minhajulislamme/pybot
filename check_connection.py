import logging
import requests
import sys
import time
import hmac
import hashlib
from binance.client import Client
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define API host
API_HOST = "https://testnet.binancefuture.com"
WS_HOST = "wss://fstream.binancefuture.com"

def check_endpoints():
    """Check if the Binance Futures Testnet endpoints are working"""
    logger.info("Checking Binance Futures Testnet endpoints...")
    
    # Test WebSocket endpoint
    ws_url = f"{WS_HOST}/ws"
    logger.info(f"WebSocket URL: {ws_url}")
    
    # Test REST API endpoints
    rest_url = f"{API_HOST}/fapi/v1/ping"
    
    try:
        response = requests.get(rest_url)
        if response.status_code == 200:
            logger.info(f"REST API endpoint is working: {rest_url}")
        else:
            logger.error(f"REST API endpoint returned status code {response.status_code}: {rest_url}")
    except Exception as e:
        logger.error(f"Error testing REST API endpoint: {e}")
    
    # Test with the API client
    try:
        client = Client(config.TEST_API_KEY, config.TEST_API_SECRET, testnet=True)
        # Test API connection
        ping = client.futures_ping()
        logger.info("API client connected successfully")
        
        # Test getting server time
        server_time = client.futures_time()
        logger.info(f"Server time: {server_time}")
        
        # Check account access using direct API call (v2 endpoint)
        try:
            url = f"{API_HOST}/fapi/v2/account"
            params = {'timestamp': int(time.time() * 1000)}
            
            # Generate signature
            query_string = '&'.join([f"{key}={params[key]}" for key in params])
            signature = hmac.new(
                config.TEST_API_SECRET.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            # Make the request
            headers = {'X-MBX-APIKEY': config.TEST_API_KEY}
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                logger.info("Successfully accessed account information via v2 endpoint")
                logger.info(f"Account data: {response.json()}")
            else:
                logger.error(f"Error accessing account info via v2 endpoint: {response.text}")
        except Exception as e:
            logger.error(f"Error accessing account info via direct API: {e}")
            
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")

if __name__ == "__main__":
    check_endpoints()
    logger.info("Connection check completed")
