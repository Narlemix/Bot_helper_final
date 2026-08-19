"""FastAPI entrypoint. Wiring only — the actual logic lives in:

    classifier.py  — free text -> request category (+ typeahead suggestions)
    dialog.py       — the conversation state machine
    mailer.py        — sending the finished request by email
    models.py        — shared data structures
    text_utils.py    — normalization / validators

data/faq.json is the single source of truth for what categories exist, their
keywords/examples, their recipient email, and the fields each one collects —
see README.md → "Добавление нового сценария" to add one without touching code.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classifier import Classifier
from .dialog import Bot
from .logging_config import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger("hrbot.main")

BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_PATH = BASE_DIR / "data" / "faq.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

APP_VERSION = "3.1.0"  # bump this on every deploy — /api/health and the startup log line both show it

classifier = Classifier(FAQ_PATH)
bot = Bot(classifier)

logger.info(
    "hrbot starting: version=%s intents=%d dry_run_email=%s",
    APP_VERSION,
    len(classifier.intents),
    os.getenv("DRY_RUN_EMAIL", "true"),
)

app = FastAPI(title="HR Helper — Обращения в кадровую и административную службу", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "intent_count": len(classifier.intents),
        "dry_run_email": os.getenv("DRY_RUN_EMAIL", "true").lower() in {"1", "true", "yes", "y", "да"},
    }


@app.get("/api/intents")
def list_intents() -> dict:
    """Lists every configured request category and its routing address.

    Handy for confirming what's actually deployed (e.g. after a redeploy)
    without needing to open faq.json on the server.
    """
    return {
        "intents": [
            {"key": key, "name": intent.name, "recipient": intent.recipient, "field_count": len(intent.fields)}
            for key, intent in classifier.intents.items()
        ]
    }


@app.get("/api/suggest")
def suggest(q: str = "") -> dict:
    return {"suggestions": classifier.suggest(q)}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    session_id = payload.session_id or str(uuid.uuid4())
    result = bot.reply(session_id, payload.message)
    result["session_id"] = session_id
    return result
