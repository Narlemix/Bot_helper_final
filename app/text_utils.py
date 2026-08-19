from __future__ import annotations

import re
from datetime import datetime

_SKIP_WORDS = {
    "нет", "неа", "нету", "не имеется", "не надо", "не нужно",
    "пропустить", "пропуск", "skip", "-", "—", "none", "n/a", "na",
}

_DATE_PATTERNS = [
    re.compile(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2}|\d{4})$"),
]

_MONEY_RE = re.compile(r"\d")


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру, убирает пунктуацию и лишние пробелы."""
    if not text:
        return ""
    t = text.strip().lower().replace("ё", "е")
    t = re.sub(r"[^0-9a-zа-я\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_skip(text: str) -> bool:
    """Проверяет, означает ли ответ пользователя «пропустить это поле»."""
    return normalize(text) in _SKIP_WORDS


def validate_date(text: str) -> str | None:
    """Парсит дату (ДД.ММ.ГГГГ и близкие форматы) в канонический вид, иначе None."""
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
    """Проверяет, что в ответе есть хотя бы одна цифра (сумма), иначе None."""
    candidate = text.strip()
    if not candidate:
        return None
    if not _MONEY_RE.search(candidate):
        return None
    return candidate


def validate_text(text: str) -> str | None:
    """Отклоняет пустой/пробельный ответ для текстового поля."""
    candidate = text.strip()
    return candidate if candidate else None
