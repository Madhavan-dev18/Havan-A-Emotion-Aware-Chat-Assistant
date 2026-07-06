"""LLM Service: Groq integration with Crisis Interceptor and Rule-Based Fallback."""
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

EMPATHETIC_RESPONSES = {
    ("sadness", "negative"): "I hear how difficult this is for you right now. I'm here to listen.",
    ("anger", "negative"): "It makes complete sense that you'd feel frustrated or angry about this.",
    ("fear", "negative"): "That sounds really overwhelming. Let's take a deep breath together.",
    ("joy", "positive"): "That's wonderful! It sounds like this brought you a lot of happiness.",
    ("surprise", "neutral"): "Oh wow, that sounds very unexpected! How are you processing it?",
    ("neutral", "neutral"): "I understand. Could you tell me a little more about what's on your mind?",
}

def generate_response(user_message, emotion_data, conversation_history, groq_api_key, groq_model):
    # ── CRISIS INTERCEPTOR (SAFETY FIRST) ────────────────────────────────
    if emotion_data.get("is_crisis"):
        return {
            "content": (
                "I'm detecting that you might be in a crisis. Please know you are not alone "
                "and support is available right now.\n\n"
                "**🇮🇳 India:** iCall — 9152987821 | Vandrevala Foundation — 1860-2662-345 (24/7)\n"
                "**🇺🇸 US:** 988 Suicide & Crisis Lifeline (call/text 988) | Crisis Text Line (text HOME to 741741)\n"
                "**🇬🇧 UK:** Samaritans — 116 123 (24/7, free)\n"
                "**🌍 International:** https://findahelpline.com\n\n"
                "If you are in immediate danger, please call your local emergency number. "
                "I am an AI, but your safety is extremely important."
            ),
            "source": "safety_interceptor"
        }

    # ── Normal LLM Flow ──────────────────────────────────────────────────
    if not groq_api_key:
        return _rule_based_fallback(emotion_data)

    try:
        client = Groq(api_key=groq_api_key)
        
        system_prompt = f"""You are Havan Vision, an empathetic AI companion.
User's text emotion: {emotion_data.get('primary_emotion')}
User's facial telemetry: {emotion_data.get('visual_emotion')}
Match your tone to their state. Be concise, supportive, and grounded."""

        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        messages.append({"role": "user", "content": user_message})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=groq_model,
            temperature=0.7,
            max_tokens=800
        )
        return {"content": chat_completion.choices[0].message.content, "source": "groq_llm"}
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return _rule_based_fallback(emotion_data)

def _rule_based_fallback(emotion_data):
    # FIXED: Dynamic dictionary lookup instead of hardcoded strings
    primary_emo = emotion_data.get("primary_emotion", "neutral")
    sentiment = emotion_data.get("sentiment", "neutral")
    
    response_text = EMPATHETIC_RESPONSES.get(
        (primary_emo, sentiment), 
        "I'm here for you. Could you tell me more about what's going on?"
    )
    return {"content": response_text, "source": "rule_fallback"}