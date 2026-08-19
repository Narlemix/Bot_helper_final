from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone

MAX_ENTRIES = 500

_buffer: deque[dict] = deque(maxlen=MAX_ENTRIES)


class BufferHandler(logging.Handler):
    """Складывает каждую лог-запись в общий кольцевой буфер (для /admin/logs)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        _buffer.append(
            {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )


def get_recent(limit: int = MAX_ENTRIES) -> list[dict]:
    """Возвращает последние записи буфера, сначала самые новые."""
    items = list(_buffer)[-limit:]
    items.reverse()
    return items
