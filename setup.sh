#!/usr/bin/env bash
# ==============================================================================
# Alpha-DQN Universal Automated Deployment & Setup Script
# ==============================================================================
# One-command deployment:
#   chmod +x setup.sh && ./setup.sh
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}       🚀 Alpha-DQN One-Command Automated Setup & Deploy        ${NC}"
echo -e "${BLUE}================================================================${NC}"

# 1. Check Python installation (>= 3.10 required)
echo -e "\n${YELLOW}[1/6] Checking Python runtime...${NC}"
PYTHON_BIN=""
for cmd in python3 python python3.11 python3.12 python3.13 python3.10; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PY_VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$("$cmd" -c 'import sys; print(sys.version_info.major)')
        PY_MINOR=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}❌ Error: Python >= 3.10 is required but not found.${NC}"
    echo "Please install Python 3.10+ (e.g., sudo apt install python3-venv python3-pip)"
    exit 1
fi
echo -e "${GREEN}✓ Found compatible Python: $PYTHON_BIN (v$PY_VER)${NC}"

# 2. Virtual Environment Setup
echo -e "\n${YELLOW}[2/6] Configuring isolated virtual environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in $SCRIPT_DIR/.venv..."
    "$PYTHON_BIN" -m venv .venv
fi

# Activate virtual environment
# shellcheck source=/dev/null
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment active: $(which python)${NC}"

# Upgrade packaging tools
pip install --quiet --upgrade pip setuptools wheel

# 3. GPU Detection & PyTorch Installation
echo -e "\n${YELLOW}[3/6] Detecting compute hardware and dependencies...${NC}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}✓ NVIDIA GPU detected!${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo -e "${BLUE}ℹ No NVIDIA GPU detected. Running on high-performance CPU mode.${NC}"
fi

echo "Installing project dependencies from requirements.txt..."
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ All dependencies installed successfully.${NC}"

# 4. Environment & Data Directory Preparation
echo -e "\n${YELLOW}[4/6] Initializing storage directories and environment config...${NC}"
mkdir -p data logs models docs

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    else
        cat << 'EOF' > .env
# Server
API_HOST=0.0.0.0
API_PORT=8001
APP_ENV=development

# Frontend
FRONTEND_URL=http://localhost:3000

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Auth
JWT_SECRET=super-secret-key-change-me-in-production
JWT_EXPIRY_HOURS=168

# Nginx Rate Limiting
NGINX_RATE_LIMIT_ZONE_SIZE=10m
NGINX_RATE_LIMIT_RATE=10r/s
NGINX_RATE_LIMIT_BURST=20
EOF
    fi
    echo -e "${GREEN}✓ Generated fresh .env file.${NC}"
else
    echo -e "${GREEN}✓ Existing .env preserved.${NC}"
fi

# 5. Automated Verification & Smoke Testing
echo -e "\n${YELLOW}[5/6] Running automated test verification suite...${NC}"
pytest tests/ -q --maxfail=1
echo -e "${GREEN}✓ All test suites passed 100%!${NC}"

# 6. Deployment Summary
echo -e "\n${YELLOW}[6/6] Finalizing deployment readiness...${NC}"

HOST=$(grep -E '^API_HOST=' .env | cut -d '=' -f2 || echo "0.0.0.0")
PORT=$(grep -E '^API_PORT=' .env | cut -d '=' -f2 || echo "8001")

TAILSCALE_IP=""
if command -v tailscale >/dev/null 2>&1; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
fi

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}        🎉 Alpha-DQN is Ready for Production Deployment!        ${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "To start the live server manually:"
echo -e "   ${BLUE}source .venv/bin/activate && python main.py${NC}"
echo -e ""
echo -e "Access URLs once started:"
echo -e "   • Localhost:       ${BLUE}http://127.0.0.1:${PORT}${NC}"
if [ -n "$TAILSCALE_IP" ]; then
echo -e "   • Tailscale VPN:   ${BLUE}http://${TAILSCALE_IP}:${PORT}${NC}"
echo -e "   • Swagger UI:      ${BLUE}http://${TAILSCALE_IP}:${PORT}/docs${NC}"
else
echo -e "   • Swagger UI:      ${BLUE}http://127.0.0.1:${PORT}/docs${NC}"
fi
echo -e "================================================================"

# If run with '--run' or 'start', start server immediately
if [ "$1" = "--run" ] || [ "$1" = "start" ]; then
    echo -e "\n🚀 Starting Alpha-DQN server on http://${HOST}:${PORT}..."
    exec python main.py
fi
