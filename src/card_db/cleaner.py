"""
Text cleaning utilities for Hearthstone card texts and names.
"""

import re

# Regex patterns for cleaning card text
_TAG_PATTERN = re.compile(r"</?[bi]>", re.IGNORECASE)
_SPECIAL_TOKEN_PATTERN = re.compile(r"(\[x\]|\$|@|_)", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0]+")
_NEWLINE_PATTERN = re.compile(r"[\r\n]+")


def clean_card_text(text: str) -> str:
    """
    Cleans raw Hearthstone text by removing markup tags, special tokens ($6, [x], _),
    and normalizing whitespace.
    """
    if not text:
        return ""

    # Remove HTML bold/italic formatting
    cleaned = _TAG_PATTERN.sub("", text)
    # Remove special engine markers: [x], $, @, _
    cleaned = _SPECIAL_TOKEN_PATTERN.sub("", cleaned)
    # Replace line breaks with a single space or period
    cleaned = _NEWLINE_PATTERN.sub(" ", cleaned)
    # Collapse multiple spaces
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)

    return cleaned.strip()
