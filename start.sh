#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start the bot processes using supervisor
sudo supervisorctl start trading_bot
sudo supervisorctl start health_monitor

# Display status
echo "Bot processes started. Checking status..."
sudo supervisorctl status

# Display log file locations
echo -e "\nLog files are available at:"
echo "/var/log/trading_bot.log"
echo "/var/log/trading_bot.err.log"
echo "/var/log/health_monitor.log"
echo "/var/log/health_monitor.err.log"

echo -e "\nTo view logs in real-time, use:"
echo "sudo tail -f /var/log/trading_bot.log"
