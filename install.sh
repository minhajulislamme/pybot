#!/bin/bash
set -e  # Exit on error

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Get current user
CURRENT_USER=$(whoami)
BOT_DIR=$(pwd)

log "Starting installation as user: $CURRENT_USER"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    log "Please run as root (use sudo)"
    exit 1
fi

# Update system packages
log "Updating system packages..."
apt-get update && apt-get upgrade -y

# Install system dependencies
log "Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3.12-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    git \
    supervisor \
    nginx \
    certbot \
    python3-certbot-nginx

# Create and prepare Python virtual environment
log "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Pre-install critical packages
log "Installing base Python packages..."
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install dependencies one by one to better handle errors
log "Installing Python packages..."
while IFS= read -r package; do
    if [[ ! -z "$package" && ! "$package" =~ ^#.*$ ]]; then
        log "Installing $package..."
        pip install --no-cache-dir "$package" || log "Warning: Failed to install $package"
    fi
done < requirements.txt

# Set up supervisor configuration
log "Configuring supervisor..."
cat > /etc/supervisor/conf.d/trading_bot.conf << EOF
[program:trading_bot]
directory=$BOT_DIR
command=$BOT_DIR/venv/bin/python trading_bot.py
user=$CURRENT_USER
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/trading_bot.err.log
stdout_logfile=/var/log/trading_bot.log
environment=PYTHONUNBUFFERED=1

[program:health_monitor]
directory=$BOT_DIR
command=$BOT_DIR/venv/bin/python health_monitor.py
user=$CURRENT_USER
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/health_monitor.err.log
stdout_logfile=/var/log/health_monitor.log
environment=PYTHONUNBUFFERED=1
EOF

# Set up log files
log "Setting up log files..."
touch /var/log/trading_bot.log /var/log/trading_bot.err.log
touch /var/log/health_monitor.log /var/log/health_monitor.err.log
chown $CURRENT_USER:$CURRENT_USER /var/log/trading_bot*
chown $CURRENT_USER:$CURRENT_USER /var/log/health_monitor*

# Make scripts executable
chmod +x start.sh stop.sh restart.sh

# Setup log rotation
cat > /etc/logrotate.d/trading_bot << EOF
/var/log/trading_bot*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $CURRENT_USER $CURRENT_USER
}
EOF

# Reload supervisor
supervisorctl reread
supervisorctl update

log "Installation complete!"
log "Use ./start.sh to start the bot"
log "Use ./stop.sh to stop the bot"
log "Use ./restart.sh to restart the bot"
