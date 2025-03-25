"""
Comprehensive test suite for the trading system.
Tests strategies, risk management, and live trading on testnet.
"""

import unittest
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any
from itertools import product

from config.config import config
from data_fetchers.binance_client import BinanceClient
from data_fetchers.websocket_client import BinanceWebsocketClient
from risk_management.risk_manager import RiskManager
from strategies.ma_crossover_strategy import MACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from backtesting.backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestTradingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize components needed for testing"""
        cls.symbol = "BTCUSDT"
        cls.timeframe = "1h"
        cls.client = BinanceClient(
            api_key=config.testnet_api_key,
            api_secret=config.testnet_api_secret,
            testnet=True
        )
        
        # Initialize risk manager
        cls.risk_manager = RiskManager(
            client=cls.client,
            max_risk_pct=1.0,
            default_stop_loss_pct=2.0,
            default_take_profit_pct=4.0,
            use_trailing_stop=True,
            max_open_positions=3
        )
        
        # Initialize strategies
        cls.strategies = {
            'MA': MACrossoverStrategy(
                client=cls.client,
                symbol=cls.symbol,
                timeframe=cls.timeframe,
                risk_manager=cls.risk_manager
            ),
            'RSI': RSIStrategy(
                client=cls.client,
                symbol=cls.symbol,
                timeframe=cls.timeframe,
                risk_manager=cls.risk_manager
            ),
            'BB': BollingerBandsStrategy(
                client=cls.client,
                symbol=cls.symbol,
                timeframe=cls.timeframe,
                risk_manager=cls.risk_manager
            )
        }
    
    def setUp(self):
        """Reset state before each test"""
        self.client.cancel_all_orders(self.symbol)
        self.client.close_all_positions(self.symbol)
    
    def test_binance_connection(self):
        """Test Binance API connection and basic functions"""
        # Test market data
        ticker = self.client.get_ticker_price(self.symbol)
        self.assertIsNotNone(ticker)
        self.assertGreater(float(ticker), 0)
        
        # Test historical data
        klines = self.client.get_historical_klines(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=100
        )
        self.assertIsNotNone(klines)
        self.assertEqual(len(klines), 100)
        
        # Test account data
        balance = self.client.get_account_balance()
        self.assertIsNotNone(balance)
    
    def test_websocket_connection(self):
        """Test WebSocket connection and data streaming"""
        ws_client = BinanceWebsocketClient(symbols=[self.symbol])
        
        # Start WebSocket
        ws_client.start()
        
        # Wait for some data
        import time
        time.sleep(5)
        
        # Verify we're receiving data
        self.assertTrue(ws_client.is_connected())
        
        # Stop WebSocket
        ws_client.stop()
    
    def test_strategy_initialization(self):
        """Test strategy initialization and indicator calculation"""
        for name, strategy in self.strategies.items():
            with self.subTest(strategy=name):
                # Test data preparation
                strategy.prepare_data()
                self.assertIsNotNone(strategy.df)
                
                # Test indicator calculation
                strategy.calculate_indicators()
                self.assertTrue(len(strategy.indicators) > 0)
    
    def test_signal_generation(self):
        """Test signal generation from each strategy"""
        for name, strategy in self.strategies.items():
            with self.subTest(strategy=name):
                strategy.prepare_data()
                signal = strategy.generate_signal()
                self.assertIn(signal, [-1, 0, 1])
    
    def test_risk_management(self):
        """Test risk management functions"""
        # Test position size calculation
        position_size = self.risk_manager.calculate_position_size(
            symbol=self.symbol,
            risk_pct=1.0,
            stop_loss_pct=2.0
        )
        self.assertGreater(position_size, 0)
        
        # Test max drawdown check
        self.assertFalse(self.risk_manager.check_max_drawdown())
    
    def test_backtesting(self):
        """Test backtesting functionality"""
        for name, strategy_class in [
            ('MA', MACrossoverStrategy),
            ('RSI', RSIStrategy),
            ('BB', BollingerBandsStrategy)
        ]:
            with self.subTest(strategy=name):
                backtester = Backtester(
                    client=self.client,
                    strategy_class=strategy_class,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                results = backtester.run()
                self.assertIsNotNone(results)
                self.assertIn('total_return', results)
    
    def test_live_trading_simulation(self):
        """Test live trading simulation on testnet"""
        # Initialize strategy
        strategy = self.strategies['MA']
        
        try:
            # Prepare data
            strategy.prepare_data()
            
            # Generate signal
            signal = strategy.generate_signal()
            
            if signal != 0:
                # Execute trade
                trade_result = self.risk_manager.execute_trade(
                    symbol=self.symbol,
                    signal=signal,
                    strategy_name='MA'
                )
                
                if trade_result:
                    # Verify trade execution
                    positions = self.client.get_open_positions()
                    self.assertTrue(any(p['symbol'] == self.symbol for p in positions))
                    
                    # Test trailing stop update
                    self.risk_manager.update_trailing_stops()
                    
                    # Close position
                    self.client.close_all_positions(self.symbol)
        
        except Exception as e:
            self.fail(f"Live trading test failed: {e}")
    
    def test_strategy_optimization(self):
        """Test strategy parameter optimization"""
        # Use non-parallel approach for testing
        backtester = Backtester(
            client=self.client,
            strategy_class=MACrossoverStrategy,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        # Define parameter ranges
        param_ranges = {
            'fast_ma_period': [10, 20],
            'slow_ma_period': [50],
            'ma_type': ['ema']
        }
        
        # Run optimization without parallel processing
        best_params = {}
        best_results = {}
        best_metric_value = float('-inf')
        metric = 'sharpe_ratio'
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(product(*param_ranges.values()))
        
        for params in param_values:
            param_dict = dict(zip(param_names, params))
            result = backtester.run(strategy_params=param_dict)
            if result and result.get(metric, float('-inf')) > best_metric_value:
                best_metric_value = result.get(metric, float('-inf'))
                best_params = param_dict
                best_results = result
        
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_results)
        self.assertGreaterEqual(len(best_params), 1)

if __name__ == '__main__':
    unittest.main()