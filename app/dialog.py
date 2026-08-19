"""The conversation itself: a small explicit state machine, one stage per turn.

Deliberately NOT written as 17 near-identical if/elif branches (one per category)
— that's exactly the kind of code that made the previous single-file version look
smaller than the actual functionality it needed to cover, and it rots the moment a
18th category is added. Instead, every category (`Intent`) declares *what data it
needs* (`Intent.fields`, defined per-category in data/faq.json) and this module
provides one generic engine that walks through those fields for whichever intent
was classified. Adding a category is a data change in faq.json, not a code change
here — see `README.md` → "Добавление нового сценария".

Stages, in order for a full "submit a request" conversation:
    new -> (disambiguate)? -> choose_action -> collecting (N fields) -> confirm -> new
"""
from __future__ import annotations

from rapidfuzz import fuzz

from .classifier import Classifier
from .mailer import send_email
from .models import Field, Intent, Session
from .text_utils import is_skip, normalize, validate_date, validate_money, validate_text

# How the engine augments each intent's own fields: every request starts with
# "who is asking" and ends with an optional free-text comment, so individual
# intents in faq.json only need to declare the fields specific to that category.
_LEADING_FIELDS = [
    Field(key="full_name", label="Ваше ФИО — это заявитель обращения.", title="ФИО заявителя", kind="text", required=True)
]
_TRAILING_FIELDS = [
    Field(
        key="comment",
        label="Есть что добавить (детали, сроки, комментарий)? Если нет — напишите «нет».",
        title="Комментарий",
        kind="text",
        required=False,
    )
]

_YES_WORDS = {"да", "отправить", "отправь", "подтверждаю", "подтвердить", "ок", "окей", "yes", "верно", "все верно", "всё верно"}
_NO_WORDS = {"нет", "отмена", "отменить", "cancel", "не надо", "стоп"}
_INFO_HINTS = ("информ", "узнать", "как ", "как?", "что нужно", "документ", "расскаж", "поясн")
_SUBMIT_HINTS = ("оформ", "обращ", "отправ", "заявк", "создат", "хочу подать", "подать")


def build_fields(intent: Intent) -> list[Field]:
    """The full ordered list of questions for this intent: name -> category-specific -> comment."""
    return [*_LEADING_FIELDS, *intent.fields, *_TRAILING_FIELDS]


class Bot:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.sessions: dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self.sessions[session_id] = session
        return session

    def reply(self, session_id: str, message: str) -> dict:
        session = self.get_session(session_id)
        message = (message or "").strip()
        if not message:
            return {"text": "Напишите, пожалуйста, ваш запрос текстом.", "state": session.stage}

        lower = normalize(message)

        if session.stage == "new":
            return self._handle_new(session, message, lower)
        if session.stage == "disambiguate":
            return self._handle_disambiguate(session, message, lower)
        if session.stage == "choose_action":
            return self._handle_choose_action(session, lower)
        if session.stage == "collecting":
            return self._handle_collecting(session, message, lower)
        if session.stage == "confirm":
            return self._handle_confirm(session, lower)

        # Defensive fallback: unknown stage somehow ended up on the session.
        session.stage = "new"
        return self._handle_new(session, message, lower)

    # ------------------------------------------------------------------ #
    # Stage handlers
    # ------------------------------------------------------------------ #
    def _handle_new(self, session: Session, message: str, lower: str) -> dict:
        best_key, score, ranked = self.classifier.classify(message)

        if best_key is None and (not ranked or ranked[0][1] < 0.30):
            example = self._sample_examples()
            return {
                "text": (
                    "Я пока не уверен, что правильно понял запрос. Попробуйте переформулировать, "
                    f"например: {example}"
                ),
                "state": "new",
            }

        if best_key is None:
            # Ambiguous: top candidates are close enough that guessing would be unreliable.
            top = ranked[:2]
            session.candidates = [key for key, _ in top]
            session.stage = "disambiguate"
            names = [self.classifier.intents[key].name for key, _ in top]
            return {
                "text": (
                    f"Похоже, речь может идти о «{names[0]}» или «{names[1]}». "
                    "Какой вариант вам нужен?"
                ),
                "state": "disambiguate",
                "options": names,
            }

        session.intent = best_key
        session.stage = "choose_action"
        intent = self.classifier.intents[best_key]
        return {
            "text": (
                f"Похоже, речь идёт о «{intent.name}».\n\n"
                "Вы хотите получить общую информацию или оформить обращение в профильную службу?"
            ),
            "state": "choose_action",
            "intent": best_key,
            "options": ["Получить информацию", "Оформить обращение"],
        }

    def _handle_disambiguate(self, session: Session, message: str, lower: str) -> dict:
        for key in session.candidates:
            intent_name_norm = normalize(self.classifier.intents[key].name)
            if intent_name_norm in lower or fuzz.WRatio(lower, intent_name_norm) >= 70:
                session.intent = key
                session.stage = "choose_action"
                session.candidates = []
                intent = self.classifier.intents[key]
                return {
                    "text": (
                        f"Хорошо, «{intent.name}». "
                        "Вы хотите получить общую информацию или оформить обращение в профильную службу?"
                    ),
                    "state": "choose_action",
                    "intent": key,
                    "options": ["Получить информацию", "Оформить обращение"],
                }
        # Didn't match either offered option. Only treat this as a brand-new
        # query if it's a *confident* classification on its own — otherwise a
        # short leftover reply (e.g. "оформить", typed out of habit) would
        # silently reclassify into an unrelated category instead of the user
        # actually picking one of the two options they were just given.
        fresh_key, fresh_score, _ = self.classifier.classify(message)
        if fresh_key is not None and fresh_score >= 0.5:
            session.candidates = []
            session.stage = "new"
            return self._handle_new(session, message, lower)
        names = [self.classifier.intents[key].name for key in session.candidates]
        return {
            "text": "Пожалуйста, выберите один из предложенных вариантов или сформулируйте вопрос точнее.",
            "state": "disambiguate",
            "options": names,
        }

    def _handle_choose_action(self, session: Session, lower: str) -> dict:
        intent = self.classifier.intents[session.intent]
        if any(hint in lower for hint in _INFO_HINTS):
            session.stage = "new"
            session.intent = None
            return {"text": intent.safe_answer, "state": "new"}
        if any(hint in lower for hint in _SUBMIT_HINTS):
            session.stage = "collecting"
            session.field_cursor = 0
            session.values = {}
            fields = build_fields(intent)
            return {"text": fields[0].label, "state": "collecting"}
        return {
            "text": "Выберите вариант: «Получить информацию» или «Оформить обращение».",
            "state": "choose_action",
            "options": ["Получить информацию", "Оформить обращение"],
        }

    def _handle_collecting(self, session: Session, message: str, lower: str) -> dict:
        intent = self.classifier.intents[session.intent]
        fields = build_fields(intent)
        current = fields[session.field_cursor]

        if not current.required and (is_skip(message) or not message.strip()):
            value: str | None = None
        else:
            value = self._validate_field(current, message)
            if value is None and current.kind != "text":
                return {"text": self._retry_prompt(current), "state": "collecting"}
            if value is None and current.required:
                return {
                    "text": f"Это поле обязательно. {current.label}",
                    "state": "collecting",
                }

        session.values[current.key] = value or ""
        session.field_cursor += 1

        if session.field_cursor < len(fields):
            return {"text": fields[session.field_cursor].label, "state": "collecting"}

        session.stage = "confirm"
        session.email_body = self.build_email(session)
        return {
            "text": self.build_summary(session),
            "state": "confirm",
            "options": ["Отправить", "Отменить"],
        }

    def _handle_confirm(self, session: Session, lower: str) -> dict:
        if lower in _YES_WORDS or any(w in lower for w in ("отправ", "подтвер")):
            intent = self.classifier.intents[session.intent]
            ok, info = send_email(intent, session.email_body or self.build_email(session))
            self._reset(session)
            if ok:
                return {"text": f"Готово! Обращение отправлено на {info}.", "state": "new"}
            return {
                "text": f"Не получилось отправить письмо автоматически ({info}). Обратитесь напрямую по адресу выше.",
                "state": "new",
            }
        if lower in _NO_WORDS or any(w in lower for w in ("отмен", "не отправ")):
            self._reset(session)
            return {"text": "Хорошо, обращение не отправлено. Если понадобится — напишите новый запрос.", "state": "new"}
        return {
            "text": "Пожалуйста, подтвердите: отправить обращение или отменить?",
            "state": "confirm",
            "options": ["Отправить", "Отменить"],
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_field(field_: Field, message: str) -> str | None:
        if field_.kind == "date":
            return validate_date(message)
        if field_.kind == "money":
            return validate_money(message)
        return validate_text(message)

    @staticmethod
    def _retry_prompt(field_: Field) -> str:
        if field_.kind == "date":
            return "Дата не распознана. Укажите её в формате ДД.ММ.ГГГГ, например 31.08.2026."
        if field_.kind == "money":
            return "Не вижу суммы в ответе. Укажите число, например: 45000 или 45 000 руб."
        return field_.label

    def _reset(self, session: Session) -> None:
        session.intent = None
        session.stage = "new"
        session.field_cursor = 0
        session.values = {}
        session.candidates = []
        session.email_body = None

    def _sample_examples(self) -> str:
        # Two short, varied example questions so the hint doesn't always look identical.
        pool = [ex for intent in self.classifier.intents.values() for ex in intent.examples[:1]]
        picks = pool[:2] if len(pool) >= 2 else pool
        return " / ".join(f"«{p}»" for p in picks) or "«как оформить увольнение сотрудника?»"

    def build_summary(self, session: Session) -> str:
        intent = self.classifier.intents[session.intent]
        fields = build_fields(intent)
        lines = [f"Проверьте данные перед отправкой ({intent.name}):", ""]
        for f in fields:
            value = session.values.get(f.key) or "—"
            lines.append(f"{f.title}: {value}")
        lines.append("")
        lines.append("Отправить обращение?")
        return "\n".join(lines)

    def build_email(self, session: Session) -> str:
        intent = self.classifier.intents[session.intent]
        fields = build_fields(intent)
        lines = [f"Обращение — {intent.name}", ""]
        for f in fields:
            value = session.values.get(f.key) or "—"
            lines.append(f"{f.title}: {value}")
        lines.append("")
        lines.append(f"Маршрут: {intent.recipient}")
        return "\n".join(lines)
