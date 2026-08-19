"""Small, dependency-free text helpers used by the classifier and the dialog engine.

Kept separate from everything else because both `classifier.py` (matching a free-form
question to an intent) and `dialog.py` (validating a typed answer to a structured
question) need the same normalization and the same "did the user just say skip / no"
detection, and duplicating that logic in two places is how it quietly drifts apart.
"""
from __future__ import annotations

import re
from datetime import datetime

# Words meaning "nothing to add" / "skip this field", used for optional fields
# (e.g. a comment field). Deliberately generous — a real user typing "неа" or
# "нету" should skip a field just as reliably as someone typing "нет".
_SKIP_WORDS = {
    "нет", "неа", "нету", "не имеется", "не надо", "не нужно",
    "пропустить", "пропуск", "skip", "-", "—", "none", "n/a", "na",
}

_DATE_PATTERNS = [
    # ДД.ММ.ГГГГ / ДД-ММ-ГГГГ / ДД/ММ/ГГГГ, with 2- or 4-digit year.
    re.compile(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2}|\d{4})$"),
]

_MONEY_RE = re.compile(r"\d")


def normalize(text: str) -> str:
    """Lowercase, collapse ё→е, strip punctuation noise, collapse whitespace.

    Used both for classifying free-form questions against known keyword/example
    phrases, and for comparing typed answers (e.g. against the skip-word list).
    Intentionally keeps Cyrillic and Latin letters/digits only, since apostrophes,
    quotes, and stray punctuation are just noise for both fuzzy matching and
    exact skip-word comparison.
    """
    if not text:
        return ""
    t = text.strip().lower().replace("ё", "е")
    t = re.sub(r"[^0-9a-zа-я\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_skip(text: str) -> bool:
    """True if the user's answer means 'nothing to add / skip this optional field'."""
    return normalize(text) in _SKIP_WORDS


def validate_date(text: str) -> str | None:
    """Parse a loosely-formatted Russian date into a canonical DD.MM.YYYY string.

    Returns None if the text doesn't look like a date at all, so the caller can
    re-prompt instead of silently storing garbage.
    """
    candidate = text.strip()
    for pattern in _DATE_PATTERNS:
        match = pattern.match(candidate)
        if not match:
            continue
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            parsed = datetime(int(year), int(month), int(day))
        except ValueError:
            return None
        return parsed.strftime("%d.%m.%Y")
    return None


def validate_money(text: str) -> str | None:
    """Loosely validate a money-ish answer (must contain at least one digit).

    Deliberately does NOT force a strict numeric format — real answers look like
    "120 000 руб", "около 85000", "45.500,00 ₽" and all of those are fine for a
    human reviewer reading the resulting email. We just guard against someone
    typing pure prose ("много") into a field meant to carry a number.
    """
    candidate = text.strip()
    if not candidate:
        return None
    if not _MONEY_RE.search(candidate):
        return None
    return candidate


def validate_text(text: str) -> str | None:
    """Minimal validator for free-text fields: reject empty/whitespace-only input."""
    candidate = text.strip()
    return candidate if candidate else None
