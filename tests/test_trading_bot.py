import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import logging
from decimal import Decimal
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_client import BinanceClient
from risk_manager import RiskManager
from trade_validator import TradeValidator
from strategy import create_strategy, SMACrossoverStrategy, RSIStrategy, MACDStrategy
from telegram_notifier import TelegramNotifier
import config

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TestTradingBot")


class TestBinanceClient(unittest.TestCase):
    """Test cases for the BinanceClient class"""
    
    @patch('binance.client.Client')
    def setUp(self, mock_client):
        """Set up test environment"""
        self.mock_binance_client = mock_client.return_value
        self.client = BinanceClient()
        
    def test_initialization(self):
        """Test client initialization"""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.test_mode, config.TEST_MODE)
    
    @patch('requests.get')
    def test_get_account_balance(self, mock_get):
        """Test getting account balance"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{'asset': 'USDT', 'balance': '1000.0'}]
        mock_get.return_value = mock_response
        
        balance = self.client.get_account_balance()
        self.assertEqual(balance, 1000.0)
    
    @patch('requests.get')
    def test_get_detailed_account_info(self, mock_get):
        """Test getting detailed account information"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'totalWalletBalance': '1000.0',
            'totalUnrealizedProfit': '50.0',
            'availableBalance': '950.0'
        }
        mock_get.return_value = mock_response
        
        account_info = self.client.get_detailed_account_info()
        self.assertEqual(account_info['totalWalletBalance'], '1000.0')
        self.assertEqual(account_info['totalUnrealizedProfit'], '50.0')
    
    @patch('requests.get')
    def test_get_all_open_positions(self, mock_get):
        """Test getting all open positions"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '0.1',
                'entryPrice': '30000',
                'markPrice': '31000',
                'unRealizedProfit': '100'
            },
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '0',  # This should be filtered out
                'entryPrice': '2000',
                'markPrice': '2000',
                'unRealizedProfit': '0'
            }
        ]
        mock_get.return_value = mock_response
        
        positions = self.client.get_all_open_positions()
        self.assertEqual(len(positions), 1)  # Only one position with non-zero amount
        self.assertEqual(positions[0]['symbol'], 'BTCUSDT')
        self.assertEqual(float(positions[0]['positionAmt']), 0.1)


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class"""
    
    @patch('binance_client.BinanceClient')
    def setUp(self, mock_client):
        """Set up test environment"""
        self.mock_client = mock_client.return_value
        self.risk_manager = RiskManager(
            self.mock_client,
            risk_percentage=1.0,
            stop_loss_percentage=2.0,
            take_profit_percentage=3.0
        )
    
    def test_initialization(self):
        """Test risk manager initialization"""
        self.assertEqual(self.risk_manager.risk_percentage, 1.0)
        self.assertEqual(self.risk_manager.stop_loss_percentage, 2.0)
        self.assertEqual(self.risk_manager.take_profit_percentage, 3.0)
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        self.mock_client.get_account_balance.return_value = 1000
        
        # Mock symbol_info for step size
        symbol_info = {
            'filters': [
                {
                    'filterType': 'LOT_SIZE',
                    'stepSize': '0.001'
                }
            ]
        }
        self.mock_client.client.get_symbol_info.return_value = symbol_info
        
        position_size = self.risk_manager.calculate_position_size(30000, 'BTCUSDT')
        
        # Expected: $1000 * 1% risk * 5x leverage / $30000 price ≈ 0.0017 BTC
        # Rounded to 3 decimal places (0.001 step size)
        expected_size = round((1000 * 0.01 * 5) / 30000, 3)
        self.assertAlmostEqual(position_size, expected_size, places=3)
    
    def test_calculate_stop_loss_price(self):
        """Test stop loss price calculation"""
        # For BUY (long) position
        entry_price = 30000
        sl_price_long = self.risk_manager.calculate_stop_loss_price(entry_price, 'BUY')
        expected_sl_long = entry_price * 0.98  # 2% below entry
        self.assertEqual(sl_price_long, expected_sl_long)
        
        # For SELL (short) position
        sl_price_short = self.risk_manager.calculate_stop_loss_price(entry_price, 'SELL')
        expected_sl_short = entry_price * 1.02  # 2% above entry
        self.assertEqual(sl_price_short, expected_sl_short)
    
    def test_calculate_take_profit_price(self):
        """Test take profit price calculation"""
        # For BUY (long) position
        entry_price = 30000
        tp_price_long = self.risk_manager.calculate_take_profit_price(entry_price, 'BUY')
        expected_tp_long = entry_price * 1.03  # 3% above entry
        self.assertEqual(tp_price_long, expected_tp_long)
        
        # For SELL (short) position
        tp_price_short = self.risk_manager.calculate_take_profit_price(entry_price, 'SELL')
        expected_tp_short = entry_price * 0.97  # 3% below entry
        self.assertEqual(tp_price_short, expected_tp_short)


class TestTradeValidator(unittest.TestCase):
    """Test cases for the TradeValidator class"""
    
    @patch('binance_client.BinanceClient')
    @patch('risk_manager.RiskManager')
    def setUp(self, mock_risk_manager, mock_client):
        """Set up test environment"""
        self.mock_client = mock_client.return_value
        self.mock_risk_manager = mock_risk_manager.return_value
        self.trade_validator = TradeValidator(self.mock_client, self.mock_risk_manager)
    
    def test_get_symbol_leverage(self):
        """Test getting symbol leverage from config"""
        # Test with a symbol defined in config
        self.assertEqual(self.trade_validator.get_symbol_leverage("BTCUSDT"), 5)
        
        # Test with a symbol not defined in config
        self.assertEqual(self.trade_validator.get_symbol_leverage("UNKNOWN"), config.DEFAULT_LEVERAGE)
    
    def test_validate_stop_loss(self):
        """Test stop loss validation"""
        entry_price = 30000
        
        # Valid long stop loss (below entry)
        sl_price_long = entry_price * 0.98
        self.assertTrue(self.trade_validator.validate_stop_loss(entry_price, sl_price_long, "BUY"))
        
        # Invalid long stop loss (above entry)
        invalid_sl_long = entry_price * 1.01
        self.assertFalse(self.trade_validator.validate_stop_loss(entry_price, invalid_sl_long, "BUY"))
        
        # Valid short stop loss (above entry)
        sl_price_short = entry_price * 1.02
        self.assertTrue(self.trade_validator.validate_stop_loss(entry_price, sl_price_short, "SELL"))
        
        # Invalid short stop loss (below entry)
        invalid_sl_short = entry_price * 0.99
        self.assertFalse(self.trade_validator.validate_stop_loss(entry_price, invalid_sl_short, "SELL"))
    
    def test_validate_take_profit(self):
        """Test take profit validation"""
        entry_price = 30000
        
        # Valid long take profit (above entry)
        tp_price_long = entry_price * 1.03
        self.assertTrue(self.trade_validator.validate_take_profit(entry_price, tp_price_long, "BUY"))
        
        # Invalid long take profit (below entry)
        invalid_tp_long = entry_price * 0.99
        self.assertFalse(self.trade_validator.validate_take_profit(entry_price, invalid_tp_long, "BUY"))
        
        # Valid short take profit (below entry)
        tp_price_short = entry_price * 0.97
        self.assertTrue(self.trade_validator.validate_take_profit(entry_price, tp_price_short, "SELL"))
        
        # Invalid short take profit (above entry)
        invalid_tp_short = entry_price * 1.01
        self.assertFalse(self.trade_validator.validate_take_profit(entry_price, invalid_tp_short, "SELL"))


class TestStrategy(unittest.TestCase):
    """Test cases for the Strategy classes"""
    
    def setUp(self):
        """Set up test environment"""
        self.symbol = "BTCUSDT"
        self.timeframe = "1h"
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create sample historical data for testing strategies"""
        # Create 100 sample candles
        timestamps = [pd.Timestamp('2023-01-01') + pd.Timedelta(hours=i) for i in range(100)]
        
        # Create price series with a trend for testing
        closes = [30000 + i * 10 + np.sin(i/5) * 100 for i in range(100)]
        highs = [c + 50 for c in closes]
        lows = [c - 50 for c in closes]
        opens = [closes[i-1] if i > 0 else closes[0] - 20 for i in range(100)]
        
        # Create a dataframe
        self.klines_data = []
        for i in range(100):
            self.klines_data.append([
                int(timestamps[i].timestamp() * 1000),  # Open time
                str(opens[i]),                         # Open
                str(highs[i]),                         # High
                str(lows[i]),                          # Low
                str(closes[i]),                        # Close
                "1000",                                # Volume
                int(timestamps[i].timestamp() * 1000) + 3600000, # Close time
                "30000000",                            # Quote asset volume
                "100",                                 # Number of trades
                "500",                                 # Taker buy base asset volume
                "15000000",                            # Taker buy quote asset volume
                "0"                                    # Ignore
            ])
    
    def test_sma_crossover_strategy(self):
        """Test SMA Crossover strategy"""
        strategy = create_strategy("SMA_CROSSOVER", self.symbol, self.timeframe, short_period=5, long_period=10)
        self.assertIsInstance(strategy, SMACrossoverStrategy)
        
        # Process data with the strategy
        df = strategy.prepare_data(self.klines_data)
        result_df = strategy.calculate_signals(df)
        
        # Check that the strategy calculated the SMAs correctly
        self.assertIn('short_sma', result_df.columns)
        self.assertIn('long_sma', result_df.columns)
        self.assertIn('signal', result_df.columns)
        
        # Check that buy/sell signals are present in the result
        signals = result_df['signal'].values
        self.assertTrue(any(signals == 1) or any(signals == -1), "No trading signals generated")
    
    def test_rsi_strategy(self):
        """Test RSI strategy"""
        strategy = create_strategy("RSI", self.symbol, self.timeframe, period=14, overbought=70, oversold=30)
        self.assertIsInstance(strategy, RSIStrategy)
        
        # Process data with the strategy
        df = strategy.prepare_data(self.klines_data)
        result_df = strategy.calculate_signals(df)
        
        # Check that the strategy calculated RSI correctly
        self.assertIn('rsi', result_df.columns)
        self.assertIn('signal', result_df.columns)
    
    def test_macd_strategy(self):
        """Test MACD strategy"""
        strategy = create_strategy("MACD", self.symbol, self.timeframe, 
                                  fast_period=12, slow_period=26, signal_period=9)
        self.assertIsInstance(strategy, MACDStrategy)
        
        # Process data with the strategy
        df = strategy.prepare_data(self.klines_data)
        result_df = strategy.calculate_signals(df)
        
        # Check that the strategy calculated MACD correctly
        self.assertIn('macd', result_df.columns)
        self.assertIn('macd_signal', result_df.columns)
        self.assertIn('macd_histogram', result_df.columns)
        self.assertIn('signal', result_df.columns)


class TestTelegramNotifier(unittest.TestCase):
    """Test cases for the TelegramNotifier class"""
    
    @patch('telegram_notifier.requests.post')
    def setUp(self, mock_post):
        """Set up test environment"""
        self.mock_post = mock_post
        self.telegram = TelegramNotifier()
    
    def test_initialization(self):
        """Test notifier initialization"""
        self.assertEqual(self.telegram.token, config.TELEGRAM_BOT_TOKEN)
        self.assertEqual(self.telegram.chat_id, config.TELEGRAM_CHAT_ID)
        self.assertEqual(self.telegram.notification_enabled, config.NOTIFICATIONS_ENABLED)
    
    @patch('telegram_notifier.requests.post')
    def test_send_message(self, mock_post):
        """Test sending a message"""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        self.telegram.send_message("Test message")
        
        # Check if the post request was made with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], f"https://api.telegram.org/bot{self.telegram.token}/sendMessage")
        self.assertEqual(kwargs['data']['chat_id'], self.telegram.chat_id)
        self.assertEqual(kwargs['data']['text'], "Test message")
    
    @patch('telegram_notifier.requests.post')
    def test_send_error_notification(self, mock_post):
        """Test sending an error notification"""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        error_msg = "Test error"
        self.telegram.send_error_notification(error_msg)
        
        # Check if the post request was made with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue(error_msg in kwargs['data']['text'])
        self.assertTrue("❌" in kwargs['data']['text'])
    
    @patch('telegram_notifier.requests.post')
    def test_disabled_notifications(self, mock_post):
        """Test that no messages are sent when notifications are disabled"""
        # Temporarily disable notifications
        original_setting = self.telegram.notification_enabled
        self.telegram.notification_enabled = False
        
        self.telegram.send_message("Test message")
        self.telegram.send_error_notification("Test error")
        
        # Verify that no requests were made
        mock_post.assert_not_called()
        
        # Restore original setting
        self.telegram.notification_enabled = original_setting


def get_test_suite():
    """Create a test suite with all test cases"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTest(loader.loadTestsFromTestCase(TestBinanceClient))
    suite.addTest(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTest(loader.loadTestsFromTestCase(TestTradeValidator))
    suite.addTest(loader.loadTestsFromTestCase(TestStrategy))
    suite.addTest(loader.loadTestsFromTestCase(TestTelegramNotifier))
    
    return suite


if __name__ == '__main__':
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = get_test_suite()
    result = runner.run(test_suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())
