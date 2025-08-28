#!/usr/bin/env bash
# VisionAssist Local Start Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD="${PYTHON_CMD:-python3}"
VENV_DIR="venv"
PORT="${PORT:-5000}"
HOST="${HOST:-127.0.0.1}"

echo -e "${GREEN}🚀 Starting VisionAssist...${NC}"

# Check Python
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not found${NC}"
    exit 1
fi

# Create/activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Activate venv
source $VENV_DIR/bin/activate

# Install/update dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}⚠️  Port $PORT is already in use${NC}"
    echo "Please stop the other process or use a different port"
    exit 1
fi

# Start the application
echo -e "${GREEN}✅ Starting server on http://$HOST:$PORT${NC}"
echo -e "${YELLOW}📸 Camera and microphone permissions may be required${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Set environment variables
export FLASK_APP=app.py
export HOST=$HOST
export PORT=$PORT
export DEBUG=false
export ML_BACKEND_ENABLED=true

# Run the app
python app.py