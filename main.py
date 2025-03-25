#!/usr/bin/env python3
"""
Enhanced main trading bot entry point with improved strategy management and periodic status reporting
"""

import logging
import time
import argparse
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from config.config import config
from data_fetchers.binance_client import BinanceClient
from data_fetchers.websocket_client import BinanceWebsocketClient
from risk_management.risk_manager import RiskManager
from strategies.ma_crossover_strategy import MACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from telegram_notifications.telegram_bot import TelegramNotifier
from backtesting.backtester import Backtester

def setup_logging():
    """Configure logging with enhanced format"""
    # Set root logger to INFO level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific loggers to appropriate levels
    logging.getLogger('websocket').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('binance').setLevel(logging.INFO)
    
    # Get the main logger
    return logging.getLogger(__name__)

def optimize_strategy(strategy_class, client, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Optimize strategy parameters using backtesting"""
    backtester = Backtester(
        client=client,
        strategy_class=strategy_class,
        symbol=symbol,
        timeframe=timeframe,
        start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Define parameter ranges for optimization
    param_ranges = {
        'MACrossoverStrategy': {
            'fast_ma_period': [10, 20, 30, 50],
            'slow_ma_period': [50, 100, 200],
            'ma_type': ['sma', 'ema']
        },
        'RSIStrategy': {
            'rsi_period': [7, 14, 21],
            'base_overbought': [70, 75, 80],
            'base_oversold': [20, 25, 30]
        },
        'BollingerBandsStrategy': {
            'bb_period': [20, 30, 40],
            'bb_std': [2.0, 2.5, 3.0],
            'volume_factor': [1.5, 2.0, 2.5],
            'trend_period': [100, 200]
        }
    }
    
    strategy_name = strategy_class.__name__
    if strategy_name in param_ranges:
        # Use fewer combinations for non-backtest mode to speed up startup
        if strategy_name == 'MACrossoverStrategy':
            param_ranges[strategy_name]['fast_ma_period'] = [20, 50]
            param_ranges[strategy_name]['slow_ma_period'] = [100, 200]
        elif strategy_name == 'BollingerBandsStrategy':
            # Reduce combinations for BollingerBandsStrategy to speed up optimization
            param_ranges[strategy_name] = {
                'bb_period': [20],  
                'bb_std': [2.0],
                'volume_factor': [2.0],
                'trend_period': [200]
            }
            
        return backtester.optimize_strategy(
            param_ranges[strategy_name],
            metric='sharpe_ratio'
        )
    return {}, {}

def generate_status_report(client, risk_manager, strategies, symbols: List[str]) -> str:
    """Generate a comprehensive status report of the trading bot"""
    report = "📊 *TRADING BOT STATUS REPORT*\n\n"
    
    # Add timestamp
    report += f"*Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add account information
    balance = client.get_account_balance()
    report += f"*Account Balance*: {balance:.2f} USDT\n"
    
    # Add trade history for the last 6 hours
    last_6h = datetime.now() - timedelta(hours=6)
    trades_history = risk_manager.open_trades
    
    report += "\n*Recent Trade History (Last 6 Hours):*\n"
    if trades_history:
        for symbol, trade in trades_history.items():
            if trade['entry_time'] >= last_6h:
                pnl = 0
                if 'exit_time' in trade:
                    pnl = trade.get('pnl', 0)
                    report += f"  - {symbol}: {trade['side']} @ {trade['entry_price']:.2f}\n"
                    report += f"    Exit: {trade['exit_price']:.2f} | PnL: {pnl:.2f} USDT\n"
                else:
                    current_price = client.get_ticker_price(symbol)
                    if current_price:
                        if trade['side'] == 'BUY':
                            pnl = (current_price - trade['entry_price']) * trade.get('quantity', 0)
                        else:
                            pnl = (trade['entry_price'] - current_price) * trade.get('quantity', 0)
                    report += f"  - {symbol}: {trade['side']} @ {trade['entry_price']:.2f}\n"
                    report += f"    Current: {current_price:.2f} | Unrealized PnL: {pnl:.2f} USDT\n"
    else:
        report += "  No trades in the last 6 hours\n"
    
    # Add open positions
    positions = client.get_open_positions()
    if positions:
        report += "\n*Open Positions:*\n"
        for pos in positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            pnl_pct = (unrealized_pnl / (abs(amt) * entry_price)) * 100 if amt != 0 else 0
            
            direction = "LONG" if amt > 0 else "SHORT"
            report += f"  - {symbol} {direction}: {abs(amt):.6f} @ {entry_price:.2f}\n"
            report += f"    Mark: {mark_price:.2f}, PnL: {unrealized_pnl:.2f} USDT ({pnl_pct:.2f}%)\n"
    else:
        report += "\nNo open positions\n"
    
    # Add recent signals for each strategy and symbol
    report += "\n*Recent Strategy Signals:*\n"
    for symbol in symbols:
        report += f"\n{symbol}:\n"
        for strategy_name, strategy in strategies[symbol].items():
            signal = strategy.generate_signal()
            signal_text = "BUY 🟢" if signal > 0 else "SELL 🔴" if signal < 0 else "NEUTRAL ⚪"
            report += f"  - {strategy_name}: {signal_text}\n"
    
    # Add market conditions
    report += "\n*Market Conditions:*\n"
    for symbol in symbols:
        price = client.get_ticker_price(symbol)
        report += f"  - {symbol}: {price:.2f} USDT\n"
    
    # Add bot uptime
    uptime = (datetime.now() - BOT_START_TIME).total_seconds()
    days, remainder = divmod(uptime, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    report += f"\n*Bot Uptime*: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
    
    return report

def execute_test_trade(client, telegram, logger):
    """Execute a small test trade to verify system functionality"""
    try:
        # Get current price and account balance
        symbol = "BTCUSDT"
        price = client.get_ticker_price(symbol)
        balance = client.get_account_balance()
        
        if not price or not balance:
            logger.error("Failed to get price or balance, aborting test trade")
            if telegram:
                telegram.send_error("Test Trade", "Failed to get market data or account balance")
            return False
        
        logger.info(f"Current {symbol} price: {price}")
        logger.info(f"Current USDT balance: {balance}")
        
        # Send notification about the test trade
        if telegram:
            telegram.send_message_sync(f"🧪 *EXECUTING TEST TRADE*\nSymbol: {symbol}\nPrice: ${price:.2f}\nBalance: ${balance:.2f}")
        
        # Calculate position size that ensures at least 100 USDT notional value
        # Use 102 USDT to be safely above the minimum
        target_notional = 102
        position_size = target_notional / float(price)
        
        # Round to 3 decimal places and ensure we're above minimum notional
        position_size = round(position_size, 3)
        trade_value = position_size * float(price)
        
        # Double check and adjust if needed
        while trade_value < 100:
            position_size = round(position_size + 0.001, 3)  # Increment by 0.001 BTC
            trade_value = position_size * float(price)
        
        # Check if we have enough balance with a 10% buffer for fees
        required_balance = trade_value * 1.1
        if balance < required_balance:
            logger.error(f"Insufficient balance ({balance:.2f} USDT) for test trade. Need at least {required_balance:.2f} USDT")
            if telegram:
                telegram.send_error("Test Trade", f"Insufficient balance ({balance:.2f} USDT) for test trade")
            return False
        
        logger.info(f"Test trade position size: {position_size} BTC (approx. ${trade_value:.2f})")
        
        # Create a market order
        order_result = client.create_market_order(
            side="BUY",
            quantity=position_size
        )
        
        if not order_result:
            logger.error("Test trade failed to execute")
            if telegram:
                telegram.send_error("Test Trade", "Failed to execute test trade on Binance")
            return False
        
        logger.info(f"Test trade executed successfully: {order_result}")
        
        # Send trade notification
        if telegram:
            telegram.send_trade_notification(
                trade_type='test',
                symbol=symbol,
                side='BUY',
                quantity=position_size,
                price=price
            )
        
        # Wait briefly to let the order settle
        time.sleep(2)
        
        # Check open positions
        positions = client.get_open_positions()
        logger.info(f"Open positions after test trade: {positions}")
        
        # Update account balance
        new_balance = client.get_account_balance()
        logger.info(f"Balance after test trade: {new_balance}")
        
        # Close the position immediately
        closed = client.close_all_positions(symbol)
        logger.info(f"Test position closing result: {closed}")
        
        # Get final balance
        final_balance = client.get_account_balance()
        logger.info(f"Final balance after closing test position: {final_balance}")
        
        # Send completion notification
        if telegram:
            telegram.send_message_sync(f"✅ *TEST TRADE COMPLETED*\nTest position for {symbol} has been closed\nSystem is functioning correctly and ready for trading")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during test trade: {e}")
        if telegram:
            telegram.send_error("Test Trade", f"Error: {str(e)}")
        return False

def send_account_info(client, telegram, logger):
    """Send account balance and positions information to Telegram"""
    try:
        # Get account balance
        balance = client.get_account_balance()
        if not balance:
            logger.error("Failed to fetch account balance")
            return False
        
        # Get open positions
        positions = client.get_open_positions()
        
        # Send account update
        if telegram:
            telegram.send_account_update(balance=balance, open_positions=positions)
            logger.info("Account information sent to Telegram")
        
        return True
    except Exception as e:
        logger.error(f"Error sending account info: {e}")
        return False

def main():
    """Enhanced main function with improved trading logic and status reporting"""
    global BOT_START_TIME
    BOT_START_TIME = datetime.now()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--mode', choices=['live', 'test', 'backtest'], default='test')
    parser.add_argument('--skip-test-trade', action='store_true', help='Skip executing test trade on startup')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting trading bot in {args.mode} mode")
    
    # Initialize components
    client = BinanceClient(
        api_key=config.api_key if args.mode == 'live' else config.testnet_api_key,
        api_secret=config.api_secret if args.mode == 'live' else config.testnet_api_secret,
        testnet=args.mode == 'test'
    )
    
    # Initialize WebSocket client for real-time data
    ws_client = BinanceWebsocketClient(symbols=config.symbols)
    
    # Initialize risk manager with conservative settings
    risk_manager = RiskManager(
        client=client,
        max_risk_pct=config.max_risk_per_trade,
        default_stop_loss_pct=config.default_stop_loss,
        default_take_profit_pct=config.default_take_profit,
        use_trailing_stop=config.use_trailing_stop,
        max_open_positions=config.max_open_positions
    )
    
    # Initialize Telegram notifier
    telegram = None
    if config.telegram_enabled:
        telegram = TelegramNotifier(
            token=config.telegram_token,
            chat_id=config.telegram_chat_id
        )
        
        # Send startup notification
        mode_emoji = "🔴" if args.mode == "live" else "🟡" if args.mode == "test" else "🔵"
        telegram.send_message_sync(f"{mode_emoji} Trading bot started in *{args.mode.upper()}* mode")
        
        # Send account information at startup
        send_account_info(client, telegram, logger)
        
        # Execute test trade to verify system functionality
        if not args.skip_test_trade:
            logger.info("Executing test trade to verify system functionality")
            execute_test_trade(client, telegram, logger)
    
    logger.info("Optimizing strategies...")
    
    # Initialize strategies with optimized parameters
    strategies = {}
    for symbol in config.symbols:
        strategies[symbol] = {}
        for Strategy in [MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy]:
            logger.info(f"Optimizing {Strategy.__name__} for {symbol}...")
            best_params, _ = optimize_strategy(Strategy, client, symbol, config.timeframes[0])
            strategies[symbol][Strategy.__name__] = Strategy(
                symbol=symbol,
                timeframe=config.timeframes[0],
                client=client,
                risk_manager=risk_manager,
                **best_params
            )
            logger.info(f"Optimized parameters for {Strategy.__name__}: {best_params}")
    
    if telegram:
        telegram.send_message_sync("✅ Strategy optimization complete")
    
    # Start WebSocket connection
    ws_client.start()
    logger.info("WebSocket connection started")
    
    # Send initial status report
    if telegram:
        initial_status = generate_status_report(client, risk_manager, strategies, config.symbols)
        telegram.send_message_sync(initial_status)
        logger.info("Initial status report sent")
    
    # Variables for status reporting
    last_status_time = datetime.now()
    status_interval = timedelta(hours=6)  # Send status every 6 hours
    
    try:
        while True:
            try:
                current_time = datetime.now()
                
                # Send periodic status report
                if (current_time - last_status_time) >= status_interval:
                    status_report = generate_status_report(client, risk_manager, strategies, config.symbols)
                    if telegram:
                        telegram.send_message_sync(status_report)
                    last_status_time = current_time
                    logger.info("Sent periodic status report")
                
                for symbol in config.symbols:
                    # Skip if price fetch fails
                    price = client.get_ticker_price(symbol)
                    if not price:
                        logger.warning(f"Failed to fetch price for {symbol}, skipping")
                        continue
                    
                    # Generate signals from all strategies
                    signals = {}
                    valid_signals = 0
                    for strategy_name, strategy in strategies[symbol].items():
                        if strategy.prepare_data():
                            strategy.calculate_indicators()
                            signals[strategy_name] = strategy.generate_signal()
                            valid_signals += 1
                        else:
                            logger.warning(f"Failed to prepare data for {strategy_name} on {symbol}, skipping")
                            signals[strategy_name] = 0
                    
                    # Skip if we don't have enough valid signals
                    if valid_signals < len(strategies[symbol]):
                        logger.warning(f"Not enough valid signals for {symbol}, skipping")
                        continue
                    
                    # Combine signals using weighted voting
                    weights = {
                        'MACrossoverStrategy': 0.3,
                        'RSIStrategy': 0.3,
                        'BollingerBandsStrategy': 0.4
                    }
                    
                    combined_signal = 0
                    for name, signal in signals.items():
                        strategy_weight = weights.get(name, 0.3)
                        combined_signal += signal * strategy_weight
                    
                    # Log all signals
                    logger.info(f"Symbol: {symbol}, Price: {price}, Signals: {signals}, Combined: {combined_signal:.2f}")
                    
                    # Execute trade if signal is strong enough
                    if abs(combined_signal) >= 0.5:  # Threshold for signal strength
                        trade_signal = 1 if combined_signal > 0 else -1
                        
                        # Double-check market volatility
                        market_ok = True
                        try:
                            market_ok = risk_manager.check_market_volatility(symbol)
                        except Exception as e:
                            logger.error(f"Error checking market volatility: {e}")
                        
                        if not market_ok:
                            logger.warning(f"Market volatility too high for {symbol}, skipping trade")
                            if telegram:
                                telegram.send_message_sync(f"⚠️ Skipped {symbol} trade due to high market volatility")
                            continue
                            
                        # Execute trade through risk manager
                        logger.info(f"Executing trade signal {trade_signal} for {symbol}")
                        trade_result = risk_manager.execute_trade(
                            symbol=symbol,
                            signal=trade_signal,
                            strategy_name="Combined"
                        )
                        
                        # Send notification if trade was executed
                        if trade_result and telegram:
                            telegram.send_trade_notification(
                                trade_type='entry',
                                symbol=symbol,
                                side="BUY" if trade_signal > 0 else "SELL",
                                quantity=trade_result.get('orders', [{}])[0].get('units', 0),
                                price=price
                            )
                            logger.info(f"Trade executed for {symbol}: {trade_result}")
                    
                    # Update trailing stops
                    updated = risk_manager.update_trailing_stops()
                    if updated > 0:
                        logger.info(f"Updated {updated} trailing stops")
                    
                    # Check for maximum drawdown
                    if risk_manager.check_max_drawdown():
                        logger.warning("Maximum drawdown reached, closing all positions")
                        closed_positions = risk_manager.close_all_positions()
                        if telegram:
                            telegram.send_message_sync(f"⚠️ Maximum drawdown reached. Closed positions: {', '.join(closed_positions)}")
                
                # Sleep between iterations
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if telegram:
                    telegram.send_error("Main Loop", str(e))
                time.sleep(60)  # Wait before retrying
                
    except KeyboardInterrupt:
        logger.info("Stopping trading bot...")
        # Close all positions before stopping
        risk_manager.close_all_positions()
        ws_client.stop()
        if telegram:
            telegram.send_message_sync("Trading bot stopped. All positions closed.")

if __name__ == "__main__":
    main()