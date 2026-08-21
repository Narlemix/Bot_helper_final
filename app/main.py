from __future__ import annotations

import logging
import os
import secrets
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classifier import Classifier
from .dialog import Bot
from .log_buffer import get_recent as get_recent_logs
from .logging_config import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger("hrbot.main")

BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_PATH = BASE_DIR / "data" / "faq.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

APP_VERSION = "3.3.0"

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

_basic_auth = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(_basic_auth)) -> None:
    """Проверяет логин/пароль для /api/admin/logs (Basic Auth)."""
    admin_user = os.getenv("ADMIN_USER")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_user or not admin_password:
        raise HTTPException(
            status_code=503,
            detail="Логи не настроены: задайте ADMIN_USER и ADMIN_PASSWORD в .env",
        )
    user_ok = secrets.compare_digest(credentials.username, admin_user)
    pass_ok = secrets.compare_digest(credentials.password, admin_password)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль", headers={"WWW-Authenticate": "Basic"})


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ResetRequest(BaseModel):
    session_id: str | None = None


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
    """Возвращает список всех тем обращений с адресами получателей."""
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


@app.post("/api/reset")
def reset(payload: ResetRequest) -> dict:
    """Завершает текущую сессию диалога и выдаёт новый session_id."""
    if payload.session_id:
        bot.sessions.pop(payload.session_id, None)
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/admin/logs")
def admin_logs_api(_: None = Depends(require_admin)) -> dict:
    return {"logs": get_recent_logs()}
