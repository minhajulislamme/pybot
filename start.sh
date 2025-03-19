#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Activate virtual environment
source venv/bin/activate

# Check if supervisor is running
if ! pgrep -x "supervisord" > /dev/null; then
    log "Starting supervisord..."
    sudo service supervisor start
fi

# Start the bot processes
log "Starting trading bot processes..."
sudo supervisorctl start trading_bot
sudo supervisorctl start health_monitor

# Wait for processes to start
sleep 5

# Check process status
STATUS=$(sudo supervisorctl status)
log "Current status:"
echo "$STATUS"

# Check if processes are running
if echo "$STATUS" | grep -q "RUNNING"; then
    log "Bot processes started successfully"
    log "Log files are available at:"
    log "- /var/log/trading_bot.log"
    log "- /var/log/trading_bot.err.log"
    log "- /var/log/health_monitor.log"
    log "- /var/log/health_monitor.err.log"
else
    log "ERROR: Failed to start bot processes"
    exit 1
fi
