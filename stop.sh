#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Stopping trading bot processes..."

# Stop processes through supervisor
sudo supervisorctl stop trading_bot
sudo supervisorctl stop health_monitor

# Wait for processes to stop
sleep 5

# Verify processes are stopped
STATUS=$(sudo supervisorctl status)
if echo "$STATUS" | grep -q "RUNNING"; then
    log "WARNING: Some processes are still running"
    echo "$STATUS"
    
    # Force stop if necessary
    log "Attempting to force stop..."
    sudo supervisorctl stop all
else
    log "All processes stopped successfully"
fi

# Clean up any remaining processes
if pgrep -f "trading_bot.py" > /dev/null; then
    log "Cleaning up remaining trading bot processes..."
    sudo pkill -f "trading_bot.py"
fi

if pgrep -f "health_monitor.py" > /dev/null; then
    log "Cleaning up remaining health monitor processes..."
    sudo pkill -f "health_monitor.py"
fi

log "Shutdown complete"
