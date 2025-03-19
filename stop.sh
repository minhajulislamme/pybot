#!/bin/bash

echo "Stopping trading bot processes..."
sudo supervisorctl stop trading_bot
sudo supervisorctl stop health_monitor

echo "Checking status..."
sudo supervisorctl status

echo -e "\nTo view final logs, use:"
echo "sudo tail /var/log/trading_bot.log"
echo "sudo tail /var/log/trading_bot.err.log"
