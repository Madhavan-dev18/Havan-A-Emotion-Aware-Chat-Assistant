"""
EmotionEngine — Multi-emotion analysis service.

Uses HuggingFace transformers (if sufficient RAM available),
otherwise falls back safely to rule-based keyword analysis.
"""

import os
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model references
_emotion_pipeline = None
_sentiment_pipeline = None
_models_attempted = False

EMOTION_COLORS = {
    "joy": "#FFD166",
    "sadness": "#4A90D9",
    "anger": "#EF476F",
    "fear": "#9B5DE5",
    "surprise": "#06D6A0",
    "disgust": "#95A3A4",
    "neutral": "#B0B0B0",
}

EMOTION_EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😰",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
}

# Crisis / distress keywords for safety layer
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self-harm",
    "harm myself", "cutting myself", "no reason to live", "hopeless",
    "can't go on", "give up on life",
]


def _load_models():
    """Attempt to load HuggingFace pipelines ONLY if explicitly allowed."""
    global _emotion_pipeline, _sentiment_pipeline, _models_attempted
    if _models_attempted:
        return
    _models_attempted = True
    
    # KILL SWITCH: Prevent OOM Crashes on 512MB Free Tier Servers
    # Unless USE_ML_MODELS is explicitly set to "true" in Render Environment, skip loading.
    if os.getenv("USE_ML_MODELS", "false").lower() != "true":
        logger.warning("ML Models disabled (USE_ML_MODELS != true). Using lightweight rule-based fallback to prevent OOM crash.")
        return

    try:
        from transformers import pipeline
        logger.info("Loading emotion model (Requires >1GB RAM)…")
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        logger.info("Loading sentiment model…")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )
        logger.info("Emotion models successfully loaded into memory.")
    except Exception as exc:
        logger.critical(f"Failed to load HuggingFace models ({exc}). Falling back to rules.")
        _emotion_pipeline = None
        _sentiment_pipeline = None


def _rule_based_emotion(text: str) -> dict:
    """Lightweight keyword-based emotion scorer."""
    text_lower = text.lower()
    keyword_map = {
        "joy": ["happy", "excited", "great", "love", "wonderful", "awesome", "glad", "pleased", "yay", "😊", "😄"],
        "sadness": ["sad", "unhappy", "depressed", "miserable", "crying", "cry", "tears", "grief", "miss", "lonely"],
        "anger": ["angry", "furious", "hate", "mad", "rage", "annoyed", "frustrated", "awful", "stupid"],
        "fear": ["scared", "afraid", "terrified", "anxious", "nervous", "worried", "panic", "dread"],
        "surprise": ["wow", "surprised", "shocked", "unexpected", "unbelievable", "omg", "wait what"],
        "disgust": ["disgusting", "gross", "horrible", "nasty", "yuck", "revolting"],
    }

    raw_counts = {}
    for emotion, keywords in keyword_map.items():
        raw_counts[emotion] = sum(1 for kw in keywords if kw in text_lower)

    total_matches = sum(raw_counts.values())

    if total_matches == 0:
        scores = {emotion: 0.0 for emotion in keyword_map}
        scores["neutral"] = 1.0
        return {"primary": "neutral", "scores": scores}

    # Build raw confidence scores per emotion.
    # Cap each emotion's raw score so a single keyword hit doesn't dominate.
    raw_scores = {}
    for emotion, count in raw_counts.items():
        raw_scores[emotion] = min(0.15 + count * 0.25, 0.95) if count > 0 else 0.0

    # Compute neutral *before* normalization:
    # fewer total keyword hits → higher neutral component.
    total_keywords = sum(len(v) for v in keyword_map.values())
    match_ratio = min(total_matches / max(total_keywords * 0.15, 1), 1.0)
    neutral_raw = max(0.05, 1.0 - match_ratio)

    # Normalize all scores (including neutral) so they sum to 1.0.
    raw_scores["neutral"] = neutral_raw
    total = sum(raw_scores.values())
    scores = {k: round(v / total, 4) for k, v in raw_scores.items()}

    primary = max(scores, key=lambda k: scores[k])
    return {"primary": primary, "scores": scores}


def _rule_based_sentiment(text: str) -> dict:
    pos = ["good", "great", "happy", "love", "wonderful", "nice", "thanks", "yes", "excited"]
    neg = ["bad", "hate", "sad", "terrible", "awful", "worst", "no", "never", "angry"]
    t = text.lower()
    p_count = sum(1 for w in pos if w in t)
    n_count = sum(1 for w in neg if w in t)
    if p_count > n_count:
        return {"label": "positive", "score": 0.6}
    if n_count > p_count:
        return {"label": "negative", "score": 0.6}
    return {"label": "neutral", "score": 0.55}


def analyze(text: str) -> dict:
    """
    Full emotion analysis pipeline.
    """
    _load_models()

    text = text.strip()
    if not text:
        return _empty_result()

    # ── Crisis detection (keyword-based, always runs) ──────────────────
    is_crisis = any(kw in text.lower() for kw in CRISIS_KEYWORDS)

    # ── Emotion scores ─────────────────────────────────────────────────
    if _emotion_pipeline is not None:
        try:
            raw = _emotion_pipeline(text[:512])[0]
            scores = {item["label"].lower(): round(item["score"], 4) for item in raw}
            primary = max(scores, key=lambda k: scores[k])
        except Exception as exc:
            logger.warning(f"Emotion pipeline failed: {exc}")
            result = _rule_based_emotion(text)
            scores = result["scores"]
            primary = result["primary"]
    else:
        result = _rule_based_emotion(text)
        scores = result["scores"]
        primary = result["primary"]

    # ── Sentiment ──────────────────────────────────────────────────────
    if _sentiment_pipeline is not None:
        try:
            raw_sent = _sentiment_pipeline(text[:512])[0]
            best = max(raw_sent, key=lambda x: x["score"])
            label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            sentiment = label_map.get(best["label"].lower(), "neutral")
            sentiment_score = round(best["score"], 4)
        except Exception as exc:
            logger.warning(f"Sentiment pipeline failed: {exc}")
            sent = _rule_based_sentiment(text)
            sentiment = sent["label"]
            sentiment_score = sent["score"]
    else:
        sent = _rule_based_sentiment(text)
        sentiment = sent["label"]
        sentiment_score = sent["score"]

    # ── Intensity: complement of neutral score ─────────────────────────
    neutral_score = scores.get("neutral", 0.5)
    intensity = round(1.0 - neutral_score, 4)

    return {
        "primary_emotion": primary,
        "emotion_scores": scores,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "intensity": intensity,
        "is_crisis": is_crisis,
        "color": EMOTION_COLORS.get(primary, "#B0B0B0"),
        "emoji": EMOTION_EMOJI.get(primary, "😐"),
    }


def _empty_result() -> dict:
    return {
        "primary_emotion": "neutral",
        "emotion_scores": {"neutral": 1.0},
        "sentiment": "neutral",
        "sentiment_score": 0.5,
        "intensity": 0.0,
        "is_crisis": False,
        "color": EMOTION_COLORS["neutral"],
        "emoji": EMOTION_EMOJI["neutral"],
    }


def get_emotion_trend(emotion_list: list[dict]) -> dict:
    """Compute trend metrics from a list of emotion analysis dicts."""
    if not emotion_list:
        return {}

    from collections import Counter
    import statistics

    primaries = [e.get("primary_emotion", "neutral") for e in emotion_list]
    intensities = [e.get("intensity", 0.0) for e in emotion_list]
    sentiments = [e.get("sentiment", "neutral") for e in emotion_list]

    emotion_counts = Counter(primaries)
    sentiment_counts = Counter(sentiments)

    # Compute trend via least-squares slope over the full intensity series
    # instead of only comparing the first and last points.
    n = len(intensities)
    if n < 2:
        trend_label = "stable"
    else:
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(intensities)
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(intensities))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0
        # Threshold: slope must exceed 0.02 per message to count as a real trend
        if slope > 0.02:
            trend_label = "rising"
        elif slope < -0.02:
            trend_label = "falling"
        else:
            trend_label = "stable"

    return {
        "dominant_emotion": emotion_counts.most_common(1)[0][0],
        "emotion_distribution": dict(emotion_counts),
        "avg_intensity": round(statistics.mean(intensities), 3),
        "intensity_trend": trend_label,
        "sentiment_distribution": dict(sentiment_counts),
        "total_messages": len(emotion_list),
    }