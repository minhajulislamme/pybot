#!/bin/bash
# Binance Futures Trading Bot - Installation Script

# Set script to exit immediately if any command fails
set -e

echo "===== Binance Futures Trading Bot - Installation Script ====="

# Get the current directory 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing from directory: $SCRIPT_DIR"

# Check if Python 3 is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing Python..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Create virtual environment if it doesn't exist
echo "Setting up Python virtual environment..."
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Create systemd service file if it doesn't exist in the system
echo "Setting up systemd service..."
SERVICE_FILE="/etc/systemd/system/binance-bot.service"

# Create the service file
cat > "$SCRIPT_DIR/binance-bot.service" << EOL
[Unit]
Description=Binance Futures Trading Bot
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python $SCRIPT_DIR/trading_bot.py
Restart=always
RestartSec=10s
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=binance-bot
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOL

# Copy service file to systemd directory
echo "Copying service file to systemd..."
sudo cp "$SCRIPT_DIR/binance-bot.service" "$SERVICE_FILE"

# Create additional scripts
echo "Creating management scripts..."

# Create start script
cat > "$SCRIPT_DIR/start.sh" << EOL
#!/bin/bash
echo "Starting Binance Futures Trading Bot..."
sudo systemctl daemon-reload
sudo systemctl start binance-bot
sudo systemctl enable binance-bot
sudo systemctl status binance-bot
EOL

# Create stop script
cat > "$SCRIPT_DIR/stop.sh" << EOL
#!/bin/bash
echo "Stopping Binance Futures Trading Bot..."
sudo systemctl stop binance-bot
sudo systemctl status binance-bot
EOL

# Create restart script
cat > "$SCRIPT_DIR/restart.sh" << EOL
#!/bin/bash
echo "Restarting Binance Futures Trading Bot..."
sudo systemctl restart binance-bot
sudo systemctl status binance-bot
EOL

# Make scripts executable
chmod +x "$SCRIPT_DIR/start.sh" "$SCRIPT_DIR/stop.sh" "$SCRIPT_DIR/restart.sh"

# Reload systemd to recognize the new service
echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "===== Installation completed ====="
echo "To start the bot, run: ./start.sh"
echo "To stop the bot, run: ./stop.sh"
echo "To restart the bot, run: ./restart.sh"
echo "To view logs, run: sudo journalctl -u binance-bot -f"