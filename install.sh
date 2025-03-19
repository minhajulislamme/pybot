#!/bin/bash

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
echo "Installing Python and dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git supervisor

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

# Set up supervisor configuration
echo "Setting up supervisor..."
sudo tee /etc/supervisor/conf.d/trading_bot.conf << EOF
[program:trading_bot]
directory=/home/minhajulislam/binance-futuer-trading-bot
command=/home/minhajulislam/binance-futuer-trading-bot/venv/bin/python trading_bot.py
user=minhajulislam
autostart=true
autorestart=true
stderr_logfile=/var/log/trading_bot.err.log
stdout_logfile=/var/log/trading_bot.log
environment=PYTHONUNBUFFERED=1

[program:health_monitor]
directory=/home/minhajulislam/binance-futuer-trading-bot
command=/home/minhajulislam/binance-futuer-trading-bot/venv/bin/python health_monitor.py
user=minhajulislam
autostart=true
autorestart=true
stderr_logfile=/var/log/health_monitor.err.log
stdout_logfile=/var/log/health_monitor.log
environment=PYTHONUNBUFFERED=1
EOF

# Create log files and set permissions
sudo touch /var/log/trading_bot.log /var/log/trading_bot.err.log
sudo touch /var/log/health_monitor.log /var/log/health_monitor.err.log
sudo chown minhajulislam:minhajulislam /var/log/trading_bot*
sudo chown minhajulislam:minhajulislam /var/log/health_monitor*

# Make scripts executable
chmod +x start.sh stop.sh

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update

echo "Installation complete! You can now run:"
echo "./start.sh - to start the bot"
echo "./stop.sh  - to stop the bot"
