"""
Risk Management module for the trading bot.
Handles position sizing, stop-loss, take-profit, and overall risk exposure.
"""

import logging
import time
import math
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, client, max_risk_pct=1.0, default_stop_loss_pct=2.0,
                default_take_profit_pct=4.0, use_trailing_stop=True, 
                max_open_positions=5, volatility_threshold=3.0,
                min_risk_reward_ratio=2.0, dynamic_position_sizing=True):
        """Initialize with additional parameters for dynamic sizing"""
        self.client = client
        self.max_risk_pct = max_risk_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.use_trailing_stop = use_trailing_stop
        self.max_open_positions = max_open_positions
        self.volatility_threshold = volatility_threshold
        self.open_trades = {}  # Keep track of open trades
        logger.info(f"Risk Manager initialized: Max Risk={max_risk_pct}%, "
                  f"Stop Loss={default_stop_loss_pct}%, Take Profit={default_take_profit_pct}%")
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.dynamic_position_sizing = dynamic_position_sizing
        self.profit_targets = [1.5, 2.0, 3.0]  # Multiple profit targets
        self.position_scale_out = [0.3, 0.3, 0.4]  # Scale out percentages
    
    def calculate_adaptive_risk(self, symbol, timeframe='1h'):
        """Calculate adaptive risk based on market conditions"""
        try:
            # Get historical data
            hist_data = self.client.get_historical_klines(timeframe=timeframe, limit=100)
            if hist_data is None or len(hist_data) < 20:
                return self.max_risk_pct
            
            # Calculate volatility
            returns = hist_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trend strength
            sma20 = hist_data['close'].rolling(window=20).mean()
            sma50 = hist_data['close'].rolling(window=50).mean()
            trend_strength = (sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
            
            # Adjust risk based on market conditions
            base_risk = self.max_risk_pct
            
            # Reduce risk in high volatility
            if volatility > self.volatility_threshold:
                base_risk *= 0.5
            
            # Increase risk in strong trends
            if abs(trend_strength) > 0.02:  # 2% trend
                base_risk *= 1.2
                
            return min(base_risk, self.max_risk_pct)  # Cap at max risk
            
        except Exception as e:
            logger.error(f"Error calculating adaptive risk: {e}")
            return self.max_risk_pct
    
    def execute_trade(self, symbol, signal, strategy_name=None, custom_risk_pct=None,
                     custom_stop_loss_pct=None, custom_take_profit_pct=None):
        """Execute trade with improved risk management"""
        if signal == 0:
            logger.debug(f"No trade signal for {symbol}")
            return None
            
        # Check market volatility before executing trade
        if not self.check_market_volatility(symbol):
            logger.warning(f"Market volatility too high for {symbol}, skipping trade")
            return None
            
        # Check if we can open a new position
        open_positions = self.client.get_open_positions()
        if len(open_positions) >= self.max_open_positions:
            logger.warning(f"Maximum open positions reached ({self.max_open_positions}), skipping trade")
            return None
        
        # Check if this symbol already has an open position
        for pos in open_positions:
            if pos['symbol'] == symbol:
                logger.info(f"Position already exists for {symbol}, updating risk parameters")
                return self._update_existing_position(symbol, pos, signal)
        
        # Get current price and account balance
        price = self.client.get_ticker_price(symbol)
        balance = self.client.get_account_balance()
        
        if price is None or balance is None:
            logger.error(f"Failed to get price or balance for {symbol}")
            return None
            
        # Determine position size based on risk
        stop_loss_pct = custom_stop_loss_pct if custom_stop_loss_pct is not None else self.default_stop_loss_pct
        take_profit_pct = custom_take_profit_pct if custom_take_profit_pct is not None else self.default_take_profit_pct
        
        # Calculate adaptive risk if dynamic sizing is enabled
        if self.dynamic_position_sizing:
            risk_pct = self.calculate_adaptive_risk(symbol)
        else:
            risk_pct = custom_risk_pct if custom_risk_pct is not None else self.max_risk_pct
            
        # Ensure risk/reward ratio meets minimum requirement
        if take_profit_pct / stop_loss_pct < self.min_risk_reward_ratio:
            logger.warning(f"Risk/reward ratio too low: {take_profit_pct/stop_loss_pct:.2f}, minimum: {self.min_risk_reward_ratio}")
            return None
            
        # Calculate multiple take-profit levels
        base_take_profit = take_profit_pct / 100
        take_profit_levels = [price * (1 + base_take_profit * target) if signal > 0 
                            else price * (1 - base_take_profit * target) 
                            for target in self.profit_targets]
        
        # Calculate position sizes for each target
        total_units = self.calculate_position_size(symbol, risk_pct, stop_loss_pct)
        target_units = [total_units * scale for scale in self.position_scale_out]
        
        # Execute main entry order
        side = "BUY" if signal > 0 else "SELL"
        orders = []
        
        for i, (units, take_profit_price) in enumerate(zip(target_units, take_profit_levels)):
            # Execute partial position
            order_result = self.client.create_market_order(symbol, side, units)
            if not order_result:
                continue
                
            # Create take profit order for this part
            take_profit_order = self.client.create_take_profit_order(
                symbol,
                "SELL" if signal > 0 else "BUY",
                units,
                take_profit_price
            )
            
            orders.append({
                'entry_order': order_result,
                'take_profit_order': take_profit_order,
                'units': units,
                'take_profit_price': take_profit_price
            })
        
        if not orders:
            logger.error("Failed to execute any orders")
            return None
            
        # Create stop loss order for entire position
        stop_loss_price = price * (1 - stop_loss_pct/100) if signal > 0 else price * (1 + stop_loss_pct/100)
        stop_loss_order = self.client.create_stop_loss_order(
            symbol,
            "SELL" if signal > 0 else "BUY",
            total_units,
            stop_loss_price
        )
        
        # Track trade info
        trade_info = {
            'symbol': symbol,
            'orders': orders,
            'stop_loss_order': stop_loss_order,
            'entry_price': price,
            'stop_loss': stop_loss_price,
            'take_profit_levels': take_profit_levels,
            'entry_time': datetime.now(),
            'strategy': strategy_name,
            'risk_pct': risk_pct,
            'trailing_stop': self.use_trailing_stop
        }
        
        self.open_trades[symbol] = trade_info
        return trade_info
    
    def _update_existing_position(self, symbol, position, new_signal):
        """
        Update an existing position based on a new signal.
        """
        try:
            position_amt = float(position['positionAmt'])
            position_side = "BUY" if position_amt > 0 else "SELL"
            
            # Convert signal to side
            signal_side = "BUY" if new_signal > 0 else "SELL"
            
            # If new signal is in the same direction, we could add to the position
            if position_side == signal_side:
                logger.info(f"New signal is in same direction as existing position for {symbol}, not adding")
                return None
            
            # If new signal is in the opposite direction, close the position
            logger.info(f"New signal is opposite of current position for {symbol}, closing position")
            
            # Cancel any pending orders first
            open_orders = self.client.get_open_orders(symbol)
            for order in open_orders:
                self.client.cancel_order(symbol, order['orderId'])
            
            # Close position with market order
            close_side = "SELL" if position_side == "BUY" else "BUY"
            units = abs(position_amt)
            
            # Execute closing order
            order_result = self.client.create_market_order(
                symbol=symbol,
                side=close_side,
                quantity=units
            )
            
            if order_result:
                # Remove from open trades tracking
                if symbol in self.open_trades:
                    del self.open_trades[symbol]
                
                logger.info(f"Successfully closed {position_side} position for {symbol}: {units} units")
                return {
                    'symbol': symbol,
                    'action': 'close_position',
                    'original_side': position_side,
                    'close_side': close_side,
                    'units': units,
                    'order_id': order_result['orderId']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            return None
    
    def update_trailing_stops(self):
        """Enhanced trailing stop management"""
        if not self.use_trailing_stop:
            return 0
            
        count = 0
        positions = self.client.get_open_positions()
        
        for position in positions:
            symbol = position['symbol']
            position_amt = float(position['positionAmt'])
            
            if symbol not in self.open_trades or position_amt == 0:
                continue
                
            trade_info = self.open_trades[symbol]
            current_price = self.client.get_ticker_price(symbol)
            
            if current_price is None:
                continue
                
            # Calculate profit percentage
            profit_pct = ((current_price - trade_info['entry_price']) / trade_info['entry_price'] * 100 
                         if position_amt > 0 else
                         (trade_info['entry_price'] - current_price) / trade_info['entry_price'] * 100)
            
            # Dynamic trailing stop based on profit
            if profit_pct >= 1.0:  # 1% profit
                trailing_pct = min(profit_pct / 2, 1.0)  # Half of current profit, max 1%
                
                new_stop = (current_price * (1 - trailing_pct/100) if position_amt > 0 
                           else current_price * (1 + trailing_pct/100))
                
                # Only update if new stop is better than current
                if ((position_amt > 0 and new_stop > trade_info['stop_loss']) or
                    (position_amt < 0 and new_stop < trade_info['stop_loss'])):
                    
                    # Cancel existing stop loss order
                    if trade_info['stop_loss_order']:
                        self.client.cancel_order(trade_info['stop_loss_order']['orderId'])
                    
                    # Create new stop loss order
                    new_stop_order = self.client.create_stop_loss_order(
                        symbol,
                        "SELL" if position_amt > 0 else "BUY",
                        abs(position_amt),
                        new_stop
                    )
                    
                    if new_stop_order:
                        logger.info(f"Updated trailing stop for {symbol}: {trade_info['stop_loss']} -> {new_stop}")
                        trade_info['stop_loss'] = new_stop
                        trade_info['stop_loss_order'] = new_stop_order
                        count += 1
        
        return count

    def calculate_position_size(self, symbol, risk_pct, stop_loss_pct):
        """
        Calculate position size based on account balance, risk percentage, and stop loss.
        
        Args:
            symbol (str): Trading symbol
            risk_pct (float): Risk percentage (of account)
            stop_loss_pct (float): Stop loss percentage (from entry price)
            
        Returns:
            float: Position size in units
        """
        price = self.client.get_ticker_price(symbol)
        balance = self.client.get_account_balance()
        
        if price is None or balance is None:
            logger.error(f"Failed to get price or balance for {symbol}")
            return 0
        
        # Calculate risk amount in USDT
        risk_amount = balance * (risk_pct / 100)
        
        # Calculate stop loss price distance
        stop_loss_distance = price * (stop_loss_pct / 100)
        
        # Calculate position size based on risk and stop loss
        if stop_loss_distance > 0:
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = 0
            logger.warning("Stop loss distance is zero, cannot calculate position size")
        
        # Round to 8 decimal places for crypto
        position_size = round(position_size, 8)
        
        logger.info(f"Calculated position size for {symbol}: {position_size} units "
                   f"(value: {position_size * price:.2f} USDT, risk: {risk_amount:.2f} USDT)")
        
        return position_size
    
    def check_max_drawdown(self, max_drawdown_pct=15.0):
        """
        Check if the account has reached the maximum allowed drawdown.
        
        Args:
            max_drawdown_pct (float): Maximum allowed drawdown percentage
            
        Returns:
            bool: True if max drawdown has been reached, False otherwise
        """
        # This would require keeping track of account balance history
        # For simplicity, we'll just check current balance vs open positions
        balance = self.client.get_account_balance()
        positions = self.client.get_open_positions()
        
        if balance is None or not positions:
            return False
        
        # Calculate total unrealized PnL
        total_pnl = 0
        for position in positions:
            entry_price = float(position.get('entryPrice', 0))
            mark_price = float(position.get('markPrice', 0))
            position_amt = float(position.get('positionAmt', 0))
            
            if position_amt > 0:  # Long position
                pnl = (mark_price - entry_price) * position_amt
            else:  # Short position
                pnl = (entry_price - mark_price) * abs(position_amt)
                
            total_pnl += pnl
        
        # Calculate drawdown
        drawdown_pct = (abs(min(0, total_pnl)) / balance) * 100
        
        if drawdown_pct > max_drawdown_pct:
            logger.warning(f"Maximum drawdown reached: {drawdown_pct:.2f}% (limit: {max_drawdown_pct}%)")
            return True
        
        return False
    
    def close_all_positions(self):
        """
        Close all open positions
        """
        try:
            positions = self.client.get_open_positions()
            closed_positions = []
            
            for position in positions:
                if float(position['positionAmt']) != 0:
                    # Determine the side needed to close the position
                    close_side = "SELL" if float(position['positionAmt']) > 0 else "BUY"
                    quantity = abs(float(position['positionAmt']))
                    
                    # Create closing order
                    order_result = self.client.create_market_order(
                        symbol=position['symbol'],
                        side=close_side,
                        quantity=quantity
                    )
                    
                    if order_result:
                        # Cancel any existing stop loss or take profit orders
                        open_orders = self.client.get_open_orders(position['symbol'])
                        for order in open_orders:
                            self.client.cancel_order(position['symbol'], order['orderId'])
                        
                        closed_positions.append(position['symbol'])
                        logger.info(f"Closed position for {position['symbol']}")
            
            return closed_positions
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return []