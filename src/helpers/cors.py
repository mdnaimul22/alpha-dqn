"""
CORS Configuration — FastAPI
==============================
Settings-driven CORS middleware. Zero hardcoding.

Usage in main.py:
    from src.api.cors import register_cors
    register_cors(app, settings)

All origins are derived from Settings (API_HOST, API_PORT, FRONTEND_URL).
"""

from fastapi.middleware.cors import CORSMiddleware


def register_cors(app, settings) -> None:
    """
    Register CORS middleware driven by Settings.

    Supports:
        - Tailscale IPs (e.g. 100.64.148.113)
        - Local development (localhost, 127.0.0.1)
        - Cross-origin browser games, SPAs, and web consoles (Binance, TradingView)
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r".*",
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["*"],
    )
