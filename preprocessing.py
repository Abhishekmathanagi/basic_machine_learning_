"""
Shared text-cleaning pipeline.

IMPORTANT: this exact sequence of steps is used both when training the
model (train_model.py) and when the Streamlit app (app.py) cleans a new
piece of text before asking the model to predict on it. If the two ever
go out of sync, predictions will be unreliable, so keep all preprocessing
logic here in one place and import it from both scripts.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_data():
    """Download the NLTK resources we need, if they aren't already present.

    Safe to call every time (e.g. on Streamlit Cloud, where the environment
    is rebuilt from scratch) because nltk.download() is a no-op if the data
    already exists locally.
    """
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            # punkt_tab doesn't exist on older NLTK versions - harmless to skip
            pass


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return "".join(ch for ch in text if not ch.isdigit())


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def remove_emoji(text: str) -> str:
    return "".join(ch for ch in text if ch.isascii())


def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words)


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline on a single string, in the same
    order used during training."""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    return text
