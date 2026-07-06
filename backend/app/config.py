import os
from datetime import timedelta

def _require_env(name: str) -> str:
    """Return the env var value or raise immediately — never fall back to a hardcoded secret."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            f"Refusing to start with a missing secret. Set it in .env or your deployment config."
        )
    return val

class Config:
    SECRET_KEY = _require_env("SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///havanvision.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", 10))

    JWT_SECRET_KEY = _require_env("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

    # Use headers instead of fragile cross-origin cookies
    JWT_TOKEN_LOCATION = ["headers"]

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 0,
    }

    def __init__(self):
        super().__init__()
        db_url = os.getenv("DATABASE_URL", "")
        if "postgresql" in db_url or "postgres" in db_url:
            self.SQLALCHEMY_ENGINE_OPTIONS = {
                **self.SQLALCHEMY_ENGINE_OPTIONS,
                "connect_args": {
                    "sslmode": "require",
                    "options": "-c statement_cache_size=0",
                },
            }

config_map = {"development": DevelopmentConfig, "production": ProductionConfig, "default": DevelopmentConfig}