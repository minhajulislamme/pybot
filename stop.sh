#!/bin/bash
echo "Stopping Binance Futures Trading Bot..."
sudo systemctl stop binance-bot
sudo systemctl status binance-bot