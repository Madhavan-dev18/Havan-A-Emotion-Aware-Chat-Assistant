"""
Havan Vision — Emotion-Aware AI Chat
Flask application factory.
"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
import os
import re

from flask_migrate import Migrate

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "60 per hour"])
migrate = Migrate()

def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__)

    # ── Proxy Fix for Rate Limiting ──────────────────────────────────────
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    from app.config import config_map
    config_obj = config_map[config_name]
    if isinstance(config_obj, type):
        config_obj = config_obj()
    app.config.from_object(config_obj)

    # ── Extensions ───────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    limiter.init_app(app)
    migrate.init_app(app, db)
    
    # ── CORS CONFIGURATION ────────────────────────────────────────────────
    # Explicit origins from env (comma-separated) — covers custom domains,
    # localhost dev, and your stable Vercel production alias.
    origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    allowed_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]

    cors_origin_regex_str = os.getenv("CORS_ORIGIN_REGEX", r"^https://havan-vision[\w-]*\.vercel\.app$")
    cors_origin_pattern = re.compile(cors_origin_regex_str)

    # supports_credentials=True is REMOVED. Headers are allowed for JWT Bearer auth.
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": allowed_origins + [cors_origin_pattern],
            }
        },
        allow_headers=["Content-Type", "Authorization"],
    )

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Blueprints ───────────────────────────────────────────────────────
    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")

    # ── Pre-warm ML Models ────────────────────────────────────────────────
    if os.getenv("USE_ML_MODELS", "false").lower() == "true":
        with app.app_context():
            try:
                from app.services.emotion_engine import _load_models
                _load_models()
                app.logger.info("ML Models pre-warmed successfully.")
            except Exception as exc:
                app.logger.critical(f"Failed to pre-warm ML Models during startup: {exc}")

    return app