# Binance Futures Trading Bot

A professional-grade trading bot for Binance Futures markets with multiple advanced strategies, real-time market data processing, risk management, and Telegram notifications.

## Features

- **API Integration**

  - Secure API authentication with Binance Futures
  - Real-time market data via WebSockets
  - Support for both live trading and testnet simulation

- **Multiple Trading Strategies**

  - Moving Average Crossover Strategy
  - RSI (Relative Strength Index) Strategy
  - Bollinger Bands Strategy
  - Dynamic strategy parameters based on market conditions

- **Advanced Risk Management**

  - Position sizing based on account balance
  - Automated stop-loss and take-profit
  - Trailing stops for maximizing profits
  - Maximum open positions and drawdown protection

- **Real-time Notifications**

  - Trade entry and exit alerts
  - Account balance and position updates
  - Strategy signals and error messages
  - All delivered via Telegram

- **Backtesting Engine**
  - Test strategies on historical data
  - Comprehensive performance metrics
  - Equity curve visualization
  - Detailed trade logs and analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- Binance account (and API keys)
- Telegram bot (optional, for notifications)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/binance-futures-bot.git
cd binance-futures-bot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create environment file:

```bash
cp .env.sample .env
```

4. Edit the `.env` file and add your API keys and settings:

```
# For live trading
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# For testnet
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here

# Trading settings
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
TRADING_TIMEFRAMES=1h,4h
DEFAULT_STRATEGY=MAcrossover

# Risk management
MAX_RISK_PER_TRADE=1.0
DEFAULT_STOP_LOSS_PCT=2.0
DEFAULT_TAKE_PROFIT_PCT=4.0

# Telegram settings (optional)
TELEGRAM_ENABLED=True
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage

### Running in Test Mode (Testnet)

This is the recommended way to start, using Binance's testnet environment:

```bash
python main.py --mode test
```

### Running in Live Mode

Once you're confident in your setup, switch to live trading:

```bash
python main.py --mode live
```

### Running Backtests

Test your strategies on historical data:

```bash
python main.py --mode backtest --backtest-symbol BTCUSDT --backtest-strategy MAcrossover --backtest-timeframe 1h --backtest-start 2023-01-01 --backtest-end 2023-03-31
```

## Available Strategies

### 1. Moving Average Crossover (MAcrossover)

Uses the crossing of a fast moving average and a slow moving average to generate buy/sell signals. The strategy buys when the fast MA crosses above the slow MA, and sells when it crosses below.

Parameters:

- `fast_ma_period`: Period for the fast moving average (default: 50)
- `slow_ma_period`: Period for the slow moving average (default: 200)
- `ma_type`: Type of moving average ('sma' or 'ema', default: 'sma')

### 2. RSI Strategy (RSI)

Uses the Relative Strength Index to identify overbought and oversold conditions. The strategy buys when RSI crosses above the oversold threshold and sells when it crosses below the overbought threshold.

Parameters:

- `rsi_period`: Period for RSI calculation (default: 14)
- `overbought`: Overbought threshold (default: 70)
- `oversold`: Oversold threshold (default: 30)

### 3. Bollinger Bands Strategy (BollingerBands)

Uses Bollinger Bands to identify market volatility and potential reversals. The strategy buys when the price touches the lower band and sells when it touches the upper band.

Parameters:

- `bb_period`: Period for Bollinger Bands calculation (default: 20)
- `bb_std_dev`: Standard deviation multiplier (default: 2.0)
- `use_squeeze`: Whether to use Bollinger Band squeeze signals (default: True)

## Risk Management Settings

The bot includes several risk management features that can be configured:

- `MAX_RISK_PER_TRADE`: Maximum percentage of account balance to risk per trade (default: 1.0%)
- `DEFAULT_STOP_LOSS_PCT`: Default stop loss percentage (default: 2.0%)
- `DEFAULT_TAKE_PROFIT_PCT`: Default take profit percentage (default: 4.0%)
- `USE_TRAILING_STOP`: Whether to use trailing stops (default: True)
- `MAX_OPEN_POSITIONS`: Maximum number of open positions allowed (default: 5)

## Project Structure

```
/py-bot/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── .env.sample                # Sample environment variables
├── config/
│   └── config.py              # Configuration management
├── data_fetchers/
│   ├── binance_client.py      # Binance API client
│   └── websocket_client.py    # WebSocket client for real-time data
├── risk_management/
│   └── risk_manager.py        # Risk management logic
├── strategies/
│   ├── base_strategy.py       # Base strategy class
│   ├── ma_crossover_strategy.py  # Moving Average Crossover strategy
│   ├── rsi_strategy.py        # RSI strategy
│   ├── bollinger_bands_strategy.py  # Bollinger Bands strategy
│   └── indicators/
│       └── technical_indicators.py  # Technical indicators
├── telegram_notifications/
│   └── telegram_bot.py        # Telegram notification system
└── backtesting/
    └── backtester.py          # Backtesting engine
```

## Security Best Practices

1. **API Key Security**:
   - Use API keys with only the necessary permissions (no withdrawal permissions required)
   - Store API keys in environment variables, never hard-code them
2. **Start with Testnet**:

   - Always test your configurations on the Binance testnet first
   - Gradually transition to live trading with small position sizes

3. **Risk Management**:
   - Start with conservative risk settings (≤1% risk per trade)
   - Regularly monitor the bot's performance and adjust risk parameters accordingly

## Disclaimer

This trading bot is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. Do not trade with money you cannot afford to lose.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
