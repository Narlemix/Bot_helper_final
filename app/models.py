from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    """Один вопрос, который бот задаёт перед оформлением обращения."""

    key: str
    label: str
    title: str
    kind: str = "text"  # "text" | "date" | "money"
    required: bool = True


@dataclass
class Intent:
    """Одна тема обращения: ключевые слова, получатель письма и нужные поля."""

    key: str
    name: str
    recipient: str
    keywords: list[str]
    examples: list[str]
    safe_answer: str
    fields: list[Field] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class Session:
    """Состояние одного диалога (одна сессия в чате)."""

    session_id: str
    intent: str | None = None
    stage: str = "new"
    field_cursor: int = 0
    values: dict[str, str] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    email_body: str | None = None
