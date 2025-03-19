import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from binance_client import BinanceClient
from strategy import create_strategy
import config

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting engine for testing trading strategies with historical data"""
    
    def __init__(self, symbol, timeframe, strategy_name, start_date, end_date, 
                 initial_balance=10000, commission=0.04):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.commission = commission / 100  # Convert percentage to decimal
        self.client = BinanceClient()
        
        # Create strategy instance
        strategy_params = {
            'short_period': config.SHORT_SMA,
            'long_period': config.LONG_SMA,
            'period': config.RSI_PERIOD,
            'overbought': config.RSI_OVERBOUGHT,
            'oversold': config.RSI_OVERSOLD
        }
        self.strategy = create_strategy(strategy_name, symbol, timeframe, **strategy_params)
        
        logger.info(f"Backtester initialized for {symbol} {timeframe} using {strategy_name} strategy")
        logger.info(f"Period: {start_date} to {end_date}, Initial Balance: {initial_balance} USDT, Commission: {commission}%")
    
    def fetch_historical_data(self):
        """Fetch historical data for the specified period"""
        try:
            # Convert dates to milliseconds timestamp
            start_timestamp = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
            
            # Map timeframe to interval string expected by Binance API
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            interval = interval_map.get(self.timeframe, Client.KLINE_INTERVAL_1HOUR)
            
            # Fetch historical klines
            klines = self.client.get_historical_klines(
                self.symbol,
                interval,
                start_timestamp,
                end_timestamp
            )
            
            logger.info(f"Fetched {len(klines)} historical candles for {self.symbol}")
            
            # Convert to DataFrame
            df = self.strategy.prepare_data(klines)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def run_backtest(self):
        """Run the backtest with the specified strategy"""
        try:
            # Fetch historical data
            df = self.fetch_historical_data()
            
            # Generate signals using the strategy
            df = self.strategy.calculate_signals(df)
            
            # Simulate trades
            df = self.simulate_trades(df)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(df)
            
            # Plot results
            self.plot_results(df)
            
            return metrics, df
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
    
    def simulate_trades(self, data):
        """Simulate trades based on signal data"""
        df = data.copy()
        
        # Initialize columns
        df['position'] = 0  # 1 for long, -1 for short, 0 for no position
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['trade_return'] = 0.0
        df['cumulative_return'] = 0.0
        df['equity'] = self.initial_balance
        
        current_position = 0
        entry_price = 0
        
        # Simulate trades row by row
        for i in range(1, len(df)):
            # Get current signal
            signal = df.iloc[i]['signal']
            price = df.iloc[i]['close']
            
            # Update position based on signal
            if current_position == 0 and signal == 1:  # Enter long
                current_position = 1
                entry_price = price
                df.iloc[i, df.columns.get_loc('position')] = current_position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                
            elif current_position == 0 and signal == -1:  # Enter short
                current_position = -1
                entry_price = price
                df.iloc[i, df.columns.get_loc('position')] = current_position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                
            elif current_position == 1 and signal == -1:  # Exit long and enter short
                # Calculate return for closed position
                trade_return = ((price / entry_price) - 1) * config.LEVERAGE
                # Subtract commission
                trade_return -= self.commission * 2  # Entry and exit commission
                
                df.iloc[i, df.columns.get_loc('exit_price')] = price
                df.iloc[i, df.columns.get_loc('trade_return')] = trade_return
                
                # Enter new short position
                current_position = -1
                entry_price = price
                df.iloc[i, df.columns.get_loc('position')] = current_position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                
            elif current_position == -1 and signal == 1:  # Exit short and enter long
                # Calculate return for closed position
                trade_return = ((entry_price / price) - 1) * config.LEVERAGE
                # Subtract commission
                trade_return -= self.commission * 2  # Entry and exit commission
                
                df.iloc[i, df.columns.get_loc('exit_price')] = price
                df.iloc[i, df.columns.get_loc('trade_return')] = trade_return
                
                # Enter new long position
                current_position = 1
                entry_price = price
                df.iloc[i, df.columns.get_loc('position')] = current_position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
            
            # Update equity
            if df.iloc[i]['trade_return'] != 0:
                # Update equity based on trade return
                previous_equity = df.iloc[i-1]['equity']
                new_equity = previous_equity * (1 + df.iloc[i]['trade_return'])
                df.iloc[i, df.columns.get_loc('equity')] = new_equity
            else:
                # Copy previous equity if no trade
                df.iloc[i, df.columns.get_loc('equity')] = df.iloc[i-1]['equity']
        
        # Calculate cumulative returns
        for i in range(1, len(df)):
            if df.iloc[i]['trade_return'] != 0:
                prev_cum_return = df.iloc[i-1]['cumulative_return']
                df.iloc[i, df.columns.get_loc('cumulative_return')] = (1 + prev_cum_return) * (1 + df.iloc[i]['trade_return']) - 1
            else:
                df.iloc[i, df.columns.get_loc('cumulative_return')] = df.iloc[i-1]['cumulative_return']
        
        return df
    
    def calculate_performance_metrics(self, df):
        """Calculate performance metrics from backtest results"""
        # Extract relevant data
        equity = df['equity'].dropna()
        returns = df['trade_return'].dropna()
        
        # Skip if no trades were made
        if len(returns) == 0:
            logger.warning("No trades were made during the backtest period")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        # Calculate total return
        total_return = (df['equity'].iloc[-1] / self.initial_balance - 1) * 100
        
        # Calculate annualized Sharpe ratio (assuming 252 trading days per year)
        risk_free_rate = 0
        daily_returns = equity.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)
        
        # Calculate maximum drawdown
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        winning_trades = returns[returns > 0]
        win_rate = len(winning_trades) / len(returns) * 100 if len(returns) > 0 else 0
        
        # Calculate total number of trades
        total_trades = len(returns)
        
        # Return metrics as dictionary
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        logger.info(f"Backtest Results: {metrics}")
        return metrics
    
    def plot_results(self, df):
        """Plot backtest results"""
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot equity curve
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['equity'])
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity (USDT)')
            plt.grid(True)
            
            # Plot cumulative returns
            plt.subplot(3, 1, 2)
            plt.plot(df.index, df['cumulative_return'] * 100)
            plt.title('Cumulative Returns (%)')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True)
            
            # Plot trades
            plt.subplot(3, 1, 3)
            plt.plot(df.index, df['close'])
            
            # Buy signals
            buys = df[df['signal'] == 1]
            plt.scatter(buys.index, buys['close'], color='green', label='Buy Signal', marker='^', alpha=1)
            
            # Sell signals
            sells = df[df['signal'] == -1]
            plt.scatter(sells.index, sells['close'], color='red', label='Sell Signal', marker='v', alpha=1)
            
            plt.title(f'{self.symbol} Price and Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.symbol}_{self.strategy_name}_backtest.png")
            logger.info(f"Backtest results plot saved as {self.symbol}_{self.strategy_name}_backtest.png")
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise


# Function to run a backtest from CLI
def run_backtest_from_config():
    # Get parameters from config
    symbol = config.SYMBOL
    timeframe = config.BACKTEST_TIMEFRAME
    strategy_name = config.STRATEGY
    start_date = config.BACKTEST_START_DATE
    end_date = config.BACKTEST_END_DATE
    
    # Create and run backtester
    backtester = Backtester(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date
    )
    
    metrics, df = backtester.run_backtest()
    
    # Print results
    print("\n==== Backtest Results ====")
    print(f"Symbol: {symbol}")
    print(f"Strategy: {strategy_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    
    return metrics, df


if __name__ == "__main__":
    from binance.client import Client
    logging.basicConfig(level=logging.INFO)
    run_backtest_from_config()