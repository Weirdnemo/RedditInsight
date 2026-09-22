"""Shared sentiment/emotion logic and credential lookup.

Kept separate from reddit_insight.py (the Streamlit app) so that standalone
scripts — fetch_sample_data.py, generate_synthetic_samples.py — can import
this without triggering the app's UI code, which only makes sense inside a
running `streamlit run` process.
"""

import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

EMOTION_KEYWORDS = {
    "joy": {"happy", "joy", "excited", "great", "wonderful", "amazing", "love", "loved", "best"},
    "sadness": {"sad", "unhappy", "depressed", "miserable", "terrible", "awful", "hate", "hated", "worst"},
    "anger": {"angry", "mad", "furious", "annoyed", "irritated", "rage"},
    "fear": {"afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"},
    "surprise": {"surprised", "shocked", "amazed", "astonished", "unexpected"},
}

STOPWORDS_EXTRA = {
    "http", "https", "www", "com", "org", "net", "imgur", "jpg", "png", "gif", "webp",
    "amp", "reddit", "redd", "edit", "deleted", "removed", "the", "and", "that", "this",
    "but", "they", "have", "from", "what", "when", "where", "which", "who", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "can", "will", "just", "should", "now",
}

_vader = None


def get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _vader = SentimentIntensityAnalyzer()
    return _vader


def analyze_comment(body: str) -> dict:
    """Run TextBlob + VADER + keyword-emotion analysis on one comment body."""
    body = body or ""
    blob = TextBlob(body)
    vscores = get_vader().polarity_scores(body)
    lowered = body.lower()
    emotion_scores = {e: sum(1 for kw in kws if kw in lowered) for e, kws in EMOTION_KEYWORDS.items()}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get) if any(emotion_scores.values()) else "neutral"
    return {
        "sentiment": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "vader_compound": vscores["compound"],
        "vader_pos": vscores["pos"],
        "vader_neg": vscores["neg"],
        "vader_neu": vscores["neu"],
        "dominant_emotion": dominant_emotion,
    }


def get_credential(name: str, default: str = "") -> str:
    """Read a credential from Streamlit secrets first, then env vars. Never hardcoded."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)
