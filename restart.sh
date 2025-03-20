#!/bin/bash
echo "Restarting Binance Futures Trading Bot..."
sudo systemctl restart binance-bot
sudo systemctl status binance-bot