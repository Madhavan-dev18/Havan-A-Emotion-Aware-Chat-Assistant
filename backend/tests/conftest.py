"""Shared test fixtures for Havan Vision backend."""
import os
import pytest

# Force test configuration BEFORE any app imports
os.environ["FLASK_ENV"] = "development"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-bytes-long"
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-at-least-32-bytes-long"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["GROQ_API_KEY"] = ""
os.environ["USE_ML_MODELS"] = "false"

from app import create_app, db as _db, limiter


@pytest.fixture(scope="session")
def app():
    """Create the Flask application for the full test session."""
    application = create_app("development")
    application.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "JWT_SECRET_KEY": "test-jwt-secret-key-at-least-32-bytes-long",
        "SECRET_KEY": "test-secret-key-that-is-at-least-32-bytes-long",
        "GROQ_API_KEY": "",
    })
    # Disable rate limiting during tests to avoid cascading 429s
    limiter.enabled = False
    yield application


@pytest.fixture(autouse=True)
def db(app):
    """Create fresh database tables for every test, then tear down."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def auth_headers(client):
    """Register a user and return auth headers with a valid access token."""
    res = client.post("/api/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
        "display_name": "Test User",
    })
    assert res.status_code == 201
    token = res.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def refresh_headers(client):
    """Register a user and return headers with a valid refresh token."""
    res = client.post("/api/auth/register", json={
        "username": "refreshuser",
        "email": "refresh@example.com",
        "password": "password123",
    })
    assert res.status_code == 201
    token = res.get_json()["refresh_token"]
    return {"Authorization": f"Bearer {token}"}
