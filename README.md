# Alpha-DQN Backend

A production-ready FastAPI backend architecture for Alpha-DQN, scaffolded with the `human-skills` framework and conforming to canonical project standards.

## 🏗️ Architecture Overview

```
alpha-dqn/
├── main.py                         # Application entry point with Lifespan & kill_pid
├── requirements.txt                # Project dependencies
├── .env.example                    # Environment template
├── deploy/nginx/                   # Nginx reverse proxy configurations
├── scripts/                        # Deployment & maintenance scripts
├── .agents/rules/                  # Architecture & coding standards
└── src/
    ├── config/                     # Single source of truth (Settings, paths, logger)
    ├── helpers/                    # Universal utilities (CORS, middleware, retry, DB)
    ├── core/                       # Pure business & domain logic
    ├── db/                         # Database models & repositories
    ├── providers/                  # External service wrappers
    ├── schema/                     # Pydantic data schemas
    ├── services/                   # Application orchestration services
    └── routers/                    # FastAPI route handlers
```

## 🚀 Getting Started

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```

### 3. Run the Development Server
```bash
python3 main.py
```
Or via uvicorn directly:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. API Endpoints
- **Health Check:** `http://127.0.0.1:8000/health`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs` (in development mode)
