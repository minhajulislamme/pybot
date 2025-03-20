# Binance API Configuration
# Set TEST_MODE to True for testnet, False for real trading
TEST_MODE = True

# API hosts
TESTNET_API_HOST = "https://testnet.binancefuture.com"
PRODUCTION_API_HOST = "https://fapi.binance.com"

# WebSocket hosts
TESTNET_WS_HOST = "wss://stream.binancefuture.com"
PRODUCTION_WS_HOST = "wss://fstream.binance.com"

# Testnet API credentials
TEST_API_KEY = "bb0ba32b12f6188db14096d2b2e4c1bc43592b2e265b5fd2ca81d5df56316884"
TEST_API_SECRET = "7d95dcd173e0e24eef369713fa716ff04a23cf3de5ec42e0470365ab32fac237"

# Real account API credentials
REAL_API_KEY = "your_real_api_key"
REAL_API_SECRET = "your_real_api_secret"

# Trading parameters for multiple symbols
TRADING_PAIRS = [
    {
        "symbol": "BTCUSDT",
        "leverage": 5,
        "risk_percentage": 1,
        "stop_loss_percentage": 2,
        "take_profit_percentage": 3,
        "strategy": "SMA_CROSSOVER",
        "short_sma": 9,
        "long_sma": 21
    },
    {
        "symbol": "ETHUSDT",
        "leverage": 5,
        "risk_percentage": 1,
        "stop_loss_percentage": 2,
        "take_profit_percentage": 3,
        "strategy": "RSI",
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
    },
    {
        "symbol": "BNBUSDT",
        "leverage": 5,
        "risk_percentage": 1,
        "stop_loss_percentage": 2,
        "take_profit_percentage": 3,
        "strategy": "MACD",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    {
        "symbol": "SOLUSDT",
        "leverage": 5,
        "risk_percentage": 0.8,
        "stop_loss_percentage": 2.5,
        "take_profit_percentage": 4,
        "strategy": "MACD",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    }
]

# Default values for parameters not specified in the trading pairs
DEFAULT_LEVERAGE = 5
DEFAULT_RISK_PERCENTAGE = 1
DEFAULT_STOP_LOSS_PERCENTAGE = 2
DEFAULT_TAKE_PROFIT_PERCENTAGE = 3
DEFAULT_STRATEGY = "SMA_CROSSOVER"
DEFAULT_SHORT_SMA = 9
DEFAULT_LONG_SMA = 21
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_RSI_OVERSOLD = 30

# Backtesting parameters
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"
BACKTEST_TIMEFRAME = "1h"  # Available: 1m, 5m, 15m, 1h, 4h, 1d
SYMBOL = "BTCUSDT"  # Default symbol for backtesting
STRATEGY = "SMA_CROSSOVER"  # Default strategy for backtesting
SHORT_SMA = 9
LONG_SMA = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Notification Settings
NOTIFICATIONS_ENABLED = True  # Enable/disable notifications
TELEGRAM_ENABLED = True       # Keep for backward compatibility
TELEGRAM_BOT_TOKEN = "8170894475:AAHRkjWwFa9vEXj-BnoWhsiRd_oJAjZzsxM"  # Your Telegram bot token
TELEGRAM_CHAT_ID = 874994865  # Your Telegram chat ID
NOTIFICATION_TYPES = {
    'trade_entry': True,     # Notify on new trade entry
    'trade_exit': True,      # Notify on trade exit
    'stop_loss': True,       # Notify on stop loss hit
    'take_profit': True,     # Notify on take profit hit
    'error': True           # Notify on errors
}

# Logging configuration
LOG_LEVEL = "INFO"  # Available: DEBUG, INFO, WARNING, ERROR

# Maximum retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Connection timeouts
API_TIMEOUT = 10  # seconds
WEBSOCKET_TIMEOUT = 30  # seconds

# Trading restrictions
MAX_LEVERAGE = 20  # Maximum allowed leverage
MIN_TRADE_AMOUNT = 5  # Minimum trade amount in USDT
MAX_ACTIVE_POSITIONS = 5  # Maximum number of active positions

# Error handling
FATAL_ERRORS = [-2015, -1021, -1001]  # Error codes that should trigger bot shutdown
RETRY_ERRORS = [-1000, -1001, -1002]  # Error codes that should trigger retry