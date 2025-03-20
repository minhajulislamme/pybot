#!/bin/bash

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements from requirements.txt
pip install -r requirements.txt

# Create service file
sudo tee /etc/systemd/system/binance-bot.service > /dev/null <<EOL
[Unit]
Description=Binance Future Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/minhajulislam/binance-futuer-trading-bot
Environment=PATH=/home/minhajulislam/binance-futuer-trading-bot/venv/bin
ExecStart=/home/minhajulislam/binance-futuer-trading-bot/venv/bin/python test_trade.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

# Set permissions
chmod +x test_trade.py
sudo systemctl daemon-reload
sudo systemctl enable binance-bot
sudo systemctl start binance-bot

echo "Installation completed! Bot service has been started."
echo "Check status with: sudo systemctl status binance-bot"
