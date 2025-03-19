#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Restarting trading bot..."

# Stop all processes
./stop.sh

# Wait for everything to stop
sleep 5

# Start all processes
./start.sh

log "Restart completed"
