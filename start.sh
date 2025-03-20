#!/bin/bash
echo "Starting Binance Futures Trading Bot..."
sudo systemctl daemon-reload
sudo systemctl start binance-bot
sudo systemctl enable binance-bot
sudo systemctl status binance-bot