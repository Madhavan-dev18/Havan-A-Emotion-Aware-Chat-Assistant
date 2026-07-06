"""Authentication routes — register, login, refresh, profile."""

import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity
)
from app import db, bcrypt, limiter
from app.models import User

auth_bp = Blueprint("auth", __name__)

@auth_bp.post("/register")
@limiter.limit("5 per minute")
def register():
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip().lower()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        display_name = (data.get("display_name") or username).strip()

        if not username or not email or not password:
            return jsonify({"error": "username, email, and password are required"}), 400

        if len(username) < 3 or len(username) > 32:
            return jsonify({"error": "Username must be 3–32 characters"}), 400

        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already taken"}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 409

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(
            username=username,
            email=email,
            password_hash=pw_hash,
            display_name=display_name,
        )
        db.session.add(user)
        db.session.commit()

        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))

        return jsonify({
            "user": user.to_public_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"REGISTER ERROR: {e}")
        return jsonify({"error": "Registration failed due to a database error."}), 500

@auth_bp.post("/login")
@limiter.limit("5 per minute")
def login():
    try:
        data = request.get_json(silent=True) or {}
        identifier = (data.get("username") or data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not identifier or not password:
            return jsonify({"error": "credentials required"}), 400

        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        if not user or not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

        user.last_seen = datetime.now(timezone.utc)
        db.session.commit()

        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))

        return jsonify({
            "user": user.to_public_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"LOGIN ERROR: {e}")
        return jsonify({"error": "Login failed due to a database error."}), 500

@auth_bp.post("/logout")
def logout():
    # Stateless JWT design: tokens cannot be server-side invalidated.
    # The client is responsible for discarding stored tokens.
    # Tokens remain technically valid until their natural expiry.
    return jsonify({
        "msg": "Client-side logout acknowledged. Discard your stored tokens. "
               "Note: JWTs remain valid until their natural expiry (stateless design)."
    }), 200

@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    user_id = get_jwt_identity() 
    access_token = create_access_token(identity=user_id)
    return jsonify({"access_token": access_token}), 200

@auth_bp.get("/me")
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = db.session.get(User, int(user_id))
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user.to_public_dict()})

@auth_bp.patch("/me")
@jwt_required()
def update_profile():
    try:
        user_id = get_jwt_identity()
        user = db.session.get(User, int(user_id))
        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.get_json(silent=True) or {}
        if "display_name" in data:
            user.display_name = data["display_name"].strip()[:100]
        if "avatar_emoji" in data:
            user.avatar_emoji = data["avatar_emoji"][:8]
        if "preferences" in data and isinstance(data["preferences"], dict):
            user.preferences = json.dumps(data["preferences"])

        db.session.commit()
        return jsonify({"user": user.to_public_dict()})

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"UPDATE PROFILE ERROR: {e}")
        return jsonify({"error": "Profile update failed due to a database error."}), 500