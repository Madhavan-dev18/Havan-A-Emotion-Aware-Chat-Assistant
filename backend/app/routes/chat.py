"""Chat routes — session management + emotion-aware message handling."""
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from app import db
from app.models import ConversationSession, Message, User
from app.services import emotion_engine, llm_service

chat_bp = Blueprint("chat", __name__)

@chat_bp.before_request
@jwt_required()
def verify_user_exists():
    user_id = get_jwt_identity()
    if not db.session.get(User, int(user_id)):
        return jsonify({"error": "User record does not exist or has been deleted"}), 401

@chat_bp.get("/sessions")
@jwt_required()
def list_sessions():
    user_id = int(get_jwt_identity())
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)
    
    pagination = (
        ConversationSession.query
        .filter_by(user_id=user_id, is_archived=False)
        .order_by(ConversationSession.updated_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify({"sessions": [s.to_dict() for s in pagination.items], "total": pagination.total, "pages": pagination.pages})

@chat_bp.post("/sessions")
@jwt_required()
def create_session():
    user_id = int(get_jwt_identity())
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New conversation").strip()[:200]
    session = ConversationSession(user_id=user_id, title=title)
    db.session.add(session)
    db.session.commit()
    return jsonify({"session": session.to_dict()}), 201

@chat_bp.get("/sessions/<session_id>")
@jwt_required()
def get_session(session_id):
    user_id = int(get_jwt_identity())
    session = ConversationSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": session.to_dict(), "messages": [m.to_dict() for m in session.messages]})

@chat_bp.delete("/sessions/<session_id>")
@jwt_required()
def delete_session(session_id):
    user_id = int(get_jwt_identity())
    session = ConversationSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404
    session.is_archived = True
    db.session.commit()
    return jsonify({"ok": True})

@chat_bp.post("/sessions/<session_id>/messages")
@jwt_required()
def send_message(session_id):
    user_id = int(get_jwt_identity())
    session = ConversationSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    raw_visual_emotion = data.get("visual_emotion", "neutral") 

    if not content:
        return jsonify({"error": "Message content required"}), 400
    if len(content) > 1500:
        return jsonify({"error": "Message exceeds maximum length of 1500 characters"}), 413

    # SECURE ENUM VALIDATION
    VALID_EMOTIONS = {"neutral", "happy", "sad", "angry", "surprised", "fear", "joy", "sadness", "anger", "surprise", "disgust"}
    visual_emotion = raw_visual_emotion if raw_visual_emotion in VALID_EMOTIONS else "neutral"

    emotion_data = emotion_engine.analyze(content)
    emotion_data["visual_emotion"] = visual_emotion 

    # 1. Fetch previous conversation history efficiently from database
    history_messages = (
        Message.query
        .filter_by(session_id=session_id)
        .order_by(Message.created_at.desc())
        .limit(current_app.config["MAX_MEMORY_TURNS"])
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in reversed(history_messages)]

    # 2. Add and commit user message to release db connection before external API call
    user_msg = Message(
        session_id=session_id, role="user", content=content,
        primary_emotion=emotion_data["primary_emotion"], visual_emotion=visual_emotion,
        emotion_scores=json.dumps(emotion_data["emotion_scores"]),
        sentiment=emotion_data["sentiment"], sentiment_score=emotion_data["sentiment_score"],
        intensity=emotion_data["intensity"],
    )
    db.session.add(user_msg)

    session.message_count = (session.message_count or 0) + 1
    session.updated_at = datetime.now(timezone.utc)
    session.dominant_emotion = emotion_data["primary_emotion"]
    if session.message_count <= 1:
        session.title = content[:60] + ("…" if len(content) > 60 else "")
    db.session.commit()

    # 3. Call external Groq API
    llm_result = llm_service.generate_response(
        user_message=content, emotion_data=emotion_data, conversation_history=history,
        groq_api_key=current_app.config.get("GROQ_API_KEY", ""),
        groq_model=current_app.config.get("GROQ_MODEL", "llama-3.1-8b-instant"),
    )

    # 4. Save and commit assistant message in a fresh database context
    assistant_msg = Message(session_id=session_id, role="assistant", content=llm_result["content"])
    db.session.add(assistant_msg)

    session.message_count = (session.message_count or 0) + 1
    db.session.commit()

    return jsonify({
        "user_message": user_msg.to_dict(),
        "assistant_message": assistant_msg.to_dict(),
        "emotion": emotion_data,
        "response_source": llm_result["source"],
    })