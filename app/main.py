from __future__ import annotations

import json
import os
import re
import smtplib
import uuid
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
FAQ_PATH = BASE / "data" / "faq.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", value.strip()))


@dataclass
class Intent:
    key: str
    name: str
    recipient: str
    keywords: list[str]
    examples: list[str]
    safe_answer: str
    references: list[str] = field(default_factory=list)


@dataclass
class Session:
    intent: Optional[str] = None
    stage: str = "new"
    full_name: Optional[str] = None
    employee_name: Optional[str] = None
    dismissal_date: Optional[str] = None
    comment: Optional[str] = None
    email_body: Optional[str] = None


class Classifier:
    def __init__(self, faq_path: Path):
        data = json.loads(faq_path.read_text(encoding="utf-8"))
        self.intents: dict[str, Intent] = {}
        docs: list[str] = []
        self.doc_keys: list[str] = []
        suggestion_pool: list[str] = []
        seen_suggestions: set[str] = set()
        for key, item in data.items():
            # Per-intent recipient override, e.g. RECIPIENT_EMAIL_ADMIN_DISMISSAL=...
            # Falls back to the value defined in faq.json. NOTE: the old global
            # RECIPIENT_EMAIL variable is intentionally no longer applied to every
            # intent, since that used to send *all* intents (admin + vahta) to the
            # same inbox and would break routing now that there is more than one.
            env_override = os.getenv(f"RECIPIENT_EMAIL_{key.upper()}")
            intent = Intent(
                key=key,
                name=item["name"],
                recipient=env_override or item["recipient"],
                keywords=item["keywords"],
                examples=item["example_questions"],
                safe_answer=item["safe_answer"],
            )
            intent.references = [*intent.keywords, *intent.examples]
            self.intents[key] = intent
            for ref in intent.references:
                docs.append(normalize(ref))
                self.doc_keys.append(key)
                if ref not in seen_suggestions:
                    seen_suggestions.add(ref)
                    suggestion_pool.append(ref)

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs)
        # Sort so full example questions (more useful as typeahead suggestions)
        # are tried first when scores tie, keywords/phrases second.
        self.suggestion_pool = suggestion_pool

    def suggest(self, query: str, limit: int = 6) -> list[str]:
        """Typeahead suggestions for the search box, tolerant of typos.

        Combines a rapidfuzz similarity score with a prefix/word-start bonus so
        that partial words (e.g. "оформ") surface phrases that start with a
        matching word (e.g. "оформление увольнения вахтовика") even before the
        fuzzy score alone would rank them highly.
        """
        q = normalize(query)
        if len(q) < 2:
            return []
        scored: list[tuple[str, float]] = []
        for phrase in self.suggestion_pool:
            norm_phrase = normalize(phrase)
            if not norm_phrase:
                continue
            ratio = fuzz.WRatio(q, norm_phrase) / 100.0
            prefix_bonus = 0.0
            if norm_phrase.startswith(q):
                prefix_bonus = 0.35
            else:
                for word in norm_phrase.split():
                    if word.startswith(q):
                        prefix_bonus = 0.25
                        break
            score = min(1.0, ratio + prefix_bonus)
            if score >= 0.55:
                scored.append((phrase, score))
        scored.sort(key=lambda pair: (-pair[1], len(pair[0])))
        result: list[str] = []
        for phrase, _ in scored:
            if phrase not in result:
                result.append(phrase)
            if len(result) >= limit:
                break
        return result

    def classify(self, text: str) -> tuple[str | None, float, list[tuple[str, float]]]:
        query = normalize(text)
        if not query:
            return None, 0.0, []
        qv = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(qv, self.matrix)[0]
        scores: dict[str, float] = {}
        for idx, value in enumerate(cosine_scores):
            key = self.doc_keys[idx]
            scores[key] = max(scores.get(key, 0.0), float(value))
        for key, intent in self.intents.items():
            best_fuzzy = max(
                (fuzz.partial_ratio(query, normalize(ref)) / 100.0 for ref in intent.references),
                default=0.0,
            )
            scores[key] = 0.70 * scores.get(key, 0.0) + 0.30 * best_fuzzy
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked or ranked[0][1] < 0.33:
            return None, ranked[0][1] if ranked else 0.0, ranked[:3]
        return ranked[0][0], ranked[0][1], ranked[:3]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ResetRequest(BaseModel):
    session_id: str


class Bot:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.sessions: dict[str, Session] = {}
        self.lock = Lock()

    def get_session(self, session_id: str) -> Session:
        with self.lock:
            return self.sessions.setdefault(session_id, Session())

    def reset(self, session_id: str) -> None:
        with self.lock:
            self.sessions[session_id] = Session()

    def reply(self, session_id: str, message: str) -> dict:
        message = message.strip()
        if not message:
            return {"text": "Напишите вопрос, и я постараюсь определить тему.", "state": "new"}

        session = self.get_session(session_id)
        lower = message.lower()
        if lower in {"/reset", "сброс", "начать заново"}:
            self.reset(session_id)
            return {"text": "Диалог сброшен. Чем могу помочь?", "state": "new"}

        if session.stage == "new":
            key, score, alternatives = self.classifier.classify(message)
            if not key:
                return {
                    "text": "Я пока не уверен, что правильно понял запрос. Попробуйте, например: «как оформить увольнение сотрудника?»",
                    "state": "new",
                    "confidence": round(score, 3),
                    "alternatives": [
                        {"intent": k, "score": round(v, 3)} for k, v in alternatives
                    ],
                }
            session.intent = key
            session.comment = message
            intent = self.classifier.intents[key]
            session.stage = "choose_action"
            ambiguity = ""
            if len(alternatives) > 1 and alternatives[1][1] > 0.88 * score:
                ambiguity = " Я вижу несколько близких вариантов, поэтому уточню, что именно вам нужно."
            return {
                "text": f"Похоже, речь идёт о «{intent.name}».{ambiguity}\n\nВы хотите получить общую информацию или оформить обращение в кадровую службу?",
                "state": session.stage,
                "confidence": round(score, 3),
                "intent": key,
                "options": ["Получить информацию", "Оформить обращение"],
            }

        if session.stage == "choose_action":
            if any(token in lower for token in ["информа", "узнать", "как", "что нужно", "документ"]):
                intent = self.classifier.intents[session.intent or "admin_dismissal"]
                session.stage = "new"
                return {"text": intent.safe_answer, "state": "new", "intent": session.intent}
            if any(token in lower for token in ["оформ", "обращ", "отправ", "заявк"]):
                session.stage = "collect_full_name"
                return {"text": "Хорошо. Укажите ваше ФИО — это заявитель обращения.", "state": session.stage}
            return {
                "text": "Выберите вариант: «Получить информацию» или «Оформить обращение».",
                "state": session.stage,
                "options": ["Получить информацию", "Оформить обращение"],
            }

        if session.stage == "collect_full_name":
            session.full_name = message
            session.stage = "collect_employee"
            return {"text": "Теперь укажите ФИО сотрудника, которого нужно уволить.", "state": session.stage}

        if session.stage == "collect_employee":
            session.employee_name = message
            session.stage = "collect_date"
            return {"text": "Укажите планируемую дату увольнения, например 31.08.2026.", "state": session.stage}

        if session.stage == "collect_date":
            if not validate_date(message):
                return {"text": "Дата не распознана. Укажите её в формате ДД.ММ.ГГГГ, например 31.08.2026.", "state": session.stage}
            session.dismissal_date = message
            session.stage = "collect_comment"
            return {
                "text": "Есть ли дополнительные обстоятельства или комментарий для кадровой службы? Если нет, напишите «нет».",
                "state": session.stage,
            }

        if session.stage == "collect_comment":
            session.comment = None if lower == "нет" else message
            session.stage = "confirm"
            session.email_body = self.build_email(session)
            return {
                "text": self.build_summary(session),
                "state": session.stage,
                "confirm": True,
                "options": ["Отправить обращение", "Изменить данные"],
            }

        if session.stage == "confirm":
            if any(token in lower for token in ["отправ", "да", "подтвержда"]):
                sent, detail = self.send_email(session)
                if sent:
                    self.reset(session_id)
                    return {"text": f"Готово. Обращение отправлено на {detail}.", "state": "new", "sent": True}
                session.stage = "confirm"
                return {"text": f"Не удалось отправить обращение: {detail}\nПроверьте SMTP-настройки сервера.", "state": session.stage, "sent": False}
            if any(token in lower for token in ["измен", "нет", "исправ"]):
                session.stage = "collect_full_name"
                return {"text": "Хорошо. Давайте начнём заново. Укажите ваше ФИО.", "state": session.stage}
            return {"text": "Нажмите «Отправить обращение» или «Изменить данные». Если отправлять нельзя, напишите «изменить».", "state": session.stage, "confirm": True}

        self.reset(session_id)
        return {"text": "Диалог сброшен. Чем могу помочь?", "state": "new"}

    def build_summary(self, session: Session) -> str:
        intent = self.classifier.intents[session.intent or "admin_dismissal"]
        return (
            f"Проверьте данные перед отправкой ({intent.name}):\n\n"
            f"Заявитель: {session.full_name}\n"
            f"Сотрудник: {session.employee_name}\n"
            f"Дата увольнения: {session.dismissal_date}\n"
            f"Комментарий: {session.comment or '—'}\n\n"
            "Отправить обращение в кадровую службу?"
        )

    def build_email(self, session: Session) -> str:
        intent = self.classifier.intents[session.intent or "admin_dismissal"]
        return (
            f"Обращение по увольнению — {intent.name}\n\n"
            f"Заявитель: {session.full_name}\n"
            f"Увольняемый сотрудник: {session.employee_name}\n"
            f"Планируемая дата увольнения: {session.dismissal_date}\n"
            f"Комментарий: {session.comment or '—'}\n\n"
            f"Маршрут: {intent.recipient}"
        )

    def send_email(self, session: Session) -> tuple[bool, str]:
        intent = self.classifier.intents[session.intent or "admin_dismissal"]
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
            return False, "SMTP не настроен"

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = "Обращение по увольнению"
        msg.set_content(session.email_body or self.build_email(session))
        try:
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(user, password)
                smtp.send_message(msg)
            return True, recipient
        except Exception as exc:
            return False, str(exc)


classifier = Classifier(FAQ_PATH)
bot = Bot(classifier)
app = FastAPI(title="HR Helper — Увольнение", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "intent_count": len(classifier.intents), "dry_run_email": os.getenv("DRY_RUN_EMAIL", "true").lower() in {"1", "true", "yes", "y", "да"}}


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
    bot.reset(payload.session_id)
    return {"ok": True}
