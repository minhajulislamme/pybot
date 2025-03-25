"""
Enhanced backtesting module with strategy optimization and advanced metrics.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Optional, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

class Backtester:
    """Enhanced backtester class with optimization capabilities"""
    
    def __init__(self, client, strategy_class, symbol, timeframe='1h', 
                 start_date=None, end_date=None, initial_balance=10000,
                 commission_rate=0.04, use_trailing_stop=True,
                 max_drawdown_pct=15.0):
        """Initialize backtester with additional risk parameters"""
        self.client = client
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_trailing_stop = use_trailing_stop
        self.max_drawdown_pct = max_drawdown_pct
        self.max_risk_per_trade = 1.0  # Maximum risk per trade in percentage
        self.default_stop_loss_pct = 2.0  # Default stop loss percentage
        self.default_take_profit_pct = 4.0  # Default take profit percentage
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=90)
        else:
            self.start_date = pd.to_datetime(start_date)
            
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate / 100
        
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.results = {}
        
        logger.info(f"Enhanced backtester initialized for {symbol}")
    
    def fetch_data(self):
        """Fetch historical data for backtesting"""
        try:
            self.data = self.client.get_historical_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=int(self.start_date.timestamp() * 1000),
                end_time=int(self.end_date.timestamp() * 1000)
            )
            return self.data is not None
        except Exception as e:
            logger.error(f"Error fetching backtest data: {e}")
            return False
    
    def run(self, strategy_params=None):
        """Run backtest with given parameters"""
        if not self.fetch_data():
            return None
            
        try:
            # Initialize strategy
            strategy = self.strategy_class(
                client=self.client,
                symbol=self.symbol,
                timeframe=self.timeframe,
                **(strategy_params or {})
            )
            
            # Reset tracking variables
            self.trades = []
            self.equity_curve = []
            current_balance = self.initial_balance
            current_position = None
            pnl = 0  # Initialize pnl variable
            
            # Process each candle
            for i in range(len(self.data)):
                # Update strategy data
                strategy.df = self.data.iloc[:i+1]
                strategy.calculate_indicators()
                
                # Generate signal
                signal = strategy.generate_signal()
                
                # Handle open position
                if current_position:
                    # Check stop loss and take profit
                    current_price = float(self.data.iloc[i]['close'])
                    
                    if current_position['side'] == 'BUY':
                        pnl = (current_price - current_position['entry_price']) * current_position['quantity']
                    else:
                        pnl = (current_position['entry_price'] - current_price) * current_position['quantity']
                    
                    # Close position if signal is opposite or hit stop loss/take profit
                    stop_loss_hit = pnl < -(current_position['entry_value'] * self.default_stop_loss_pct / 100)
                    take_profit_hit = pnl > (current_position['entry_value'] * self.default_take_profit_pct / 100)
                    
                    if signal * current_position['signal'] < 0 or stop_loss_hit or take_profit_hit:
                        # Close position
                        current_balance += pnl - (current_position['entry_value'] * self.commission_rate)
                        self.trades.append({
                            'exit_time': self.data.index[i],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'balance': current_balance
                        })
                        current_position = None
                
                # Open new position
                elif signal != 0:
                    # Calculate position size
                    risk_amount = current_balance * (self.max_risk_per_trade / 100)
                    current_price = float(self.data.iloc[i]['close'])
                    position_size = risk_amount / current_price
                    
                    # Open position
                    current_position = {
                        'signal': signal,
                        'side': 'BUY' if signal > 0 else 'SELL',
                        'entry_price': current_price,
                        'quantity': position_size,
                        'entry_value': position_size * current_price,
                        'entry_time': self.data.index[i]
                    }
                    
                    self.trades.append({
                        'entry_time': current_position['entry_time'],
                        'entry_price': current_position['entry_price'],
                        'side': current_position['side'],
                        'quantity': current_position['quantity'],
                        'entry_value': current_position['entry_value']
                    })
                
                # Update equity curve
                self.equity_curve.append({
                    'timestamp': self.data.index[i],
                    'equity': current_balance + (pnl if current_position else 0)
                })
            
            # Close any remaining position
            if current_position:
                current_price = float(self.data.iloc[-1]['close'])
                if current_position['side'] == 'BUY':
                    pnl = (current_price - current_position['entry_price']) * current_position['quantity']
                else:
                    pnl = (current_position['entry_price'] - current_price) * current_position['quantity']
                    
                current_balance += pnl - (current_position['entry_value'] * self.commission_rate)
                self.trades.append({
                    'exit_time': self.data.index[-1],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'balance': current_balance
                })
            
            # Calculate performance metrics
            self.calculate_performance_metrics()
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def optimize_strategy(self, param_ranges: Dict[str, List[Any]], 
                         metric: str = 'sharpe_ratio',
                         max_workers: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize strategy parameters using sequential processing.
        
        Args:
            param_ranges: Dictionary of parameter names and their possible values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor')
            max_workers: Maximum number of parallel processes (ignored, kept for compatibility)
            
        Returns:
            Tuple of (best parameters, best results)
        """
        if self.data is None:
            self.fetch_data()
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(product(*param_ranges.values()))
        
        logger.info(f"Starting strategy optimization with {len(param_values)} combinations")
        
        # Run backtests sequentially to avoid multiprocessing issues
        results = []
        best_params = {}
        best_results = {}
        best_metric_value = float('-inf')
        
        for params in param_values:
            param_dict = dict(zip(param_names, params))
            result = self.run(strategy_params=param_dict)
            results.append(result)
            
            # Track best parameters
            if result and result.get(metric, float('-inf')) > best_metric_value:
                best_metric_value = result.get(metric, float('-inf'))
                best_params = param_dict
                best_results = result
        
        logger.info(f"Optimization complete. Best {metric}: {best_results.get(metric, 'N/A')}")
        return best_params, best_results
    
    def _run_backtest_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest with given parameters"""
        strategy = self.strategy_class(
            client=self.client,
            symbol=self.symbol,
            timeframe=self.timeframe,
            **params
        )
        
        return self.run(strategy_params=params)
    
    def calculate_performance_metrics(self) -> None:
        """Calculate basic performance metrics from backtest results"""
        if not self.equity_curve:
            logger.warning("No equity curve data to calculate performance metrics")
            self.results = {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'risk_adjusted_return': 0,
                'total_trades': 0
            }
            return
            
        try:
            # Get initial and final balance
            initial_balance = self.initial_balance
            final_balance = self.equity_curve[-1]['equity'] if self.equity_curve else initial_balance
            
            # Calculate returns
            total_return_abs = final_balance - initial_balance
            total_return_pct = (total_return_abs / initial_balance) * 100
            
            # Calculate trading period
            start_date = self.start_date
            end_date = self.end_date
            trading_days = (end_date - start_date).days
            trading_years = trading_days / 365
            
            # Calculate annualized return
            annualized_return = ((1 + total_return_pct/100)**(1/trading_years) - 1) * 100 if trading_years > 0 else 0
            
            # Store basic metrics
            self.results = {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return_abs': total_return_abs,
                'total_return': total_return_pct,
                'trading_days': trading_days,
                'trading_years': trading_years,
                'annualized_return': annualized_return,
                'sharpe_ratio': 0,  # Will be updated in advanced metrics
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'risk_adjusted_return': 0,
                'total_trades': len(self.trades)
            }
            
            # Calculate advanced metrics
            self.calculate_advanced_metrics()
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Set default values to avoid errors
            self.results = {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'risk_adjusted_return': 0,
                'total_trades': 0
            }
    
    def calculate_advanced_metrics(self) -> None:
        """Calculate advanced performance metrics"""
        if not self.trades or not self.equity_curve:
            return
            
        try:
            # Convert to DataFrames for easier calculation
            trades_df = pd.DataFrame(self.trades)
            equity_df = pd.DataFrame(self.equity_curve)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            # Advanced metrics
            daily_returns = equity_df.set_index('timestamp')['equity'].pct_change()
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)
            
            # Trade metrics
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate profit factor and recovery factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate average holding time
            trades_df['duration'] = pd.to_datetime(trades_df['timestamp'])
            avg_hold_time = (trades_df['duration'] - trades_df['duration'].shift()).mean()
            
            # Store results
            self.results.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'avg_hold_time': avg_hold_time,
                'risk_adjusted_return': self.results['total_return'] / max_drawdown if max_drawdown > 0 else float('inf'),
                'expectancy': (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss),
                'kelly_fraction': ((win_rate/100) - ((1-win_rate/100)/(avg_win/avg_loss if avg_loss > 0 else float('inf'))))
            })
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown and drawdown duration"""
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio using downside deviation"""
        try:
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) < 2:
                return 0
                
            downside_std = np.sqrt(np.mean(downside_returns**2))
            
            if downside_std == 0:
                return 0
                
            return (excess_returns.mean() * 252) / (downside_std * np.sqrt(252))
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def plot_advanced_analysis(self, save_path: Optional[str] = None) -> None:
        """Generate advanced analysis plots"""
        if not self.trades or not self.equity_curve:
            return
            
        try:
            trades_df = pd.DataFrame(self.trades)
            equity_df = pd.DataFrame(self.equity_curve)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot equity curve
            axes[0, 0].plot(equity_df['timestamp'], equity_df['equity'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Equity')
            axes[0, 0].grid(True)
            
            # Plot drawdown
            rolling_max = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
            axes[0, 1].fill_between(equity_df['timestamp'], drawdown, 0, color='red', alpha=0.3)
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # Plot trade distribution
            if len(trades_df) > 0:
                trades_df['pnl'].hist(ax=axes[1, 0], bins=50)
                axes[1, 0].set_title('Trade PnL Distribution')
                axes[1, 0].set_xlabel('PnL')
                axes[1, 0].set_ylabel('Frequency')
                
                # Plot monthly returns
                monthly_returns = equity_df.set_index('timestamp')['equity'].resample('M').last().pct_change()
                monthly_returns.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Monthly Returns')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Return %')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Advanced analysis plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error generating advanced analysis plots: {e}")
    
    def generate_detailed_report(self, save_path: Optional[str] = None) -> str:
        """Generate detailed performance report with advanced metrics"""
        if not self.results:
            return "No results available"
            
        report = "# Detailed Backtest Report\n\n"
        report += f"Strategy: {self.strategy_class.__name__}\n"
        report += f"Symbol: {self.symbol}\n"
        report += f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n\n"
        
        report += "## Performance Metrics\n\n"
        report += f"- Initial Balance: ${self.initial_balance:,.2f}\n"
        report += f"- Final Balance: ${self.results['final_balance']:,.2f}\n"
        report += f"- Total Return: {self.results['total_return']:.2f}%\n"
        report += f"- Annualized Return: {self.results.get('annualized_return', 0):.2f}%\n"
        report += f"- Sharpe Ratio: {self.results['sharpe_ratio']:.2f}\n"
        report += f"- Sortino Ratio: {self.results['sortino_ratio']:.2f}\n"
        report += f"- Maximum Drawdown: {self.results['max_drawdown']:.2f}%\n"
        report += f"- Risk-Adjusted Return: {self.results['risk_adjusted_return']:.2f}\n\n"
        
        report += "## Trading Statistics\n\n"
        report += f"- Total Trades: {self.results['total_trades']}\n"
        report += f"- Win Rate: {self.results['win_rate']:.2f}%\n"
        report += f"- Profit Factor: {self.results['profit_factor']:.2f}\n"
        report += f"- Average Win: ${self.results['avg_win']:,.2f}\n"
        report += f"- Average Loss: ${self.results['avg_loss']:,.2f}\n"
        report += f"- Risk-Reward Ratio: {self.results['avg_win']/self.results['avg_loss'] if self.results['avg_loss'] > 0 else float('inf'):.2f}\n"
        report += f"- Expectancy: ${self.results['expectancy']:,.2f}\n"
        report += f"- Kelly Fraction: {self.results['kelly_fraction']:.2f}\n"
        report += f"- Average Holding Time: {self.results['avg_hold_time']}\n\n"
        
        report += "## Risk Metrics\n\n"
        report += f"- Volatility (Annual): {self.results['volatility']*100:.2f}%\n"
        report += f"- Value at Risk (95%): ${self.results.get('var_95', 0):,.2f}\n"
        report += f"- Expected Shortfall: ${self.results.get('expected_shortfall', 0):,.2f}\n"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Detailed report saved to {save_path}")
        
        return report
        
    def analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade patterns and performance by various factors"""
        if not self.trades:
            return {}
            
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Convert timestamp to datetime if needed
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            analysis = {
                'time_analysis': self._analyze_time_patterns(trades_df),
                'streaks': self._analyze_win_loss_streaks(trades_df),
                'position_sizing': self._analyze_position_sizing(trades_df),
                'market_conditions': self._analyze_market_conditions(trades_df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _analyze_time_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade performance by time patterns"""
        try:
            # Add time components
            trades_df['hour'] = trades_df['timestamp'].dt.hour
            trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()
            trades_df['month'] = trades_df['timestamp'].dt.month_name()
            
            # Analyze by hour
            hourly_stats = trades_df.groupby('hour').agg({
                'pnl': ['count', 'mean', 'sum'],
                'price': 'mean'
            }).round(2)
            
            # Analyze by day of week
            daily_stats = trades_df.groupby('day_of_week').agg({
                'pnl': ['count', 'mean', 'sum'],
                'price': 'mean'
            }).round(2)
            
            # Analyze by month
            monthly_stats = trades_df.groupby('month').agg({
                'pnl': ['count', 'mean', 'sum'],
                'price': 'mean'
            }).round(2)
            
            return {
                'hourly_stats': hourly_stats.to_dict(),
                'daily_stats': daily_stats.to_dict(),
                'monthly_stats': monthly_stats.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}
    
    def _analyze_win_loss_streaks(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winning and losing streaks"""
        try:
            # Calculate trade results (1 for win, -1 for loss)
            trades_df['result'] = np.where(trades_df['pnl'] > 0, 1, -1)
            
            # Calculate streaks
            streak_changes = trades_df['result'].ne(trades_df['result'].shift())
            streak_id = streak_changes.cumsum()
            streaks = trades_df.groupby(streak_id)['result'].agg(['count', 'first'])
            
            # Calculate streak statistics
            win_streaks = streaks[streaks['first'] == 1]['count']
            loss_streaks = streaks[streaks['first'] == -1]['count']
            
            return {
                'max_win_streak': win_streaks.max() if len(win_streaks) > 0 else 0,
                'max_loss_streak': loss_streaks.max() if len(loss_streaks) > 0 else 0,
                'avg_win_streak': win_streaks.mean() if len(win_streaks) > 0 else 0,
                'avg_loss_streak': loss_streaks.mean() if len(loss_streaks) > 0 else 0,
                'streak_distribution': streaks['count'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing streaks: {e}")
            return {}
    
    def _analyze_position_sizing(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position sizing effectiveness"""
        try:
            # Calculate position size relative to account balance
            trades_df['position_size_pct'] = (trades_df['position'] * trades_df['price'] / 
                                            trades_df['balance'] * 100)
            
            # Group trades by position size quartiles
            size_quartiles = pd.qcut(trades_df['position_size_pct'], 4, labels=['Small', 'Medium', 'Large', 'Very Large'])
            size_analysis = trades_df.groupby(size_quartiles).agg({
                'pnl': ['count', 'mean', 'sum', 'std'],
                'position_size_pct': 'mean'
            }).round(2)
            
            return size_analysis.to_dict()
            
        except Exception as e:
            logger.error(f"Error analyzing position sizing: {e}")
            return {}
    
    def _analyze_market_conditions(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance under different market conditions"""
        try:
            if self.data is None:
                return {}
                
            # Calculate market conditions
            returns = self.data['close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            trend = self.data['close'].rolling(window=50).mean().pct_change()
            
            # Classify market conditions
            market_conditions = pd.DataFrame(index=self.data.index)
            market_conditions['volatility'] = pd.qcut(volatility, 3, labels=['Low', 'Medium', 'High'])
            market_conditions['trend'] = pd.qcut(trend, 3, labels=['Bearish', 'Neutral', 'Bullish'])
            
            # Merge with trades
            trades_df = trades_df.join(market_conditions, on='timestamp', how='left')
            
            # Analyze performance by market condition
            vol_analysis = trades_df.groupby('volatility').agg({
                'pnl': ['count', 'mean', 'sum', 'std']
            }).round(2)
            
            trend_analysis = trades_df.groupby('trend').agg({
                'pnl': ['count', 'mean', 'sum', 'std']
            }).round(2)
            
            return {
                'volatility_analysis': vol_analysis.to_dict(),
                'trend_analysis': trend_analysis.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}