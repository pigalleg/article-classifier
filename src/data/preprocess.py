import re

def clean_text(text: str) -> str:
    """
    Basic text cleaner for abstracts and questions.
    - Lowercase
    - Remove extra spaces, newlines, punctuation artifacts
    - Keep words/numbers only
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%/.,:;()\- ]", "", text)
    text = text.strip()
    return text
