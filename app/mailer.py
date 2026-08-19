from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from .models import Intent


def send_email(intent: Intent, body: str) -> tuple[bool, str]:
    """Отправляет письмо на адрес темы (или имитирует отправку в режиме DRY_RUN_EMAIL)."""
    recipient = intent.recipient
    dry_run = os.getenv("DRY_RUN_EMAIL", "true").lower() in {"1", "true", "yes", "y", "да"}
    if dry_run:
        return True, f"{recipient} (DRY_RUN: письмо не отправлялось)"

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or user
    if not all([host, user, password, sender]):
        return False, "SMTP не настроен (проверьте SMTP_HOST/SMTP_USER/SMTP_PASSWORD в .env)"

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = f"Обращение — {intent.name}"
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        return True, recipient
    except Exception as exc:
        return False, f"ошибка отправки: {exc}"
