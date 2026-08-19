from __future__ import annotations

import logging

from rapidfuzz import fuzz

from .classifier import Classifier
from .mailer import send_email
from .models import Field, Intent, Session
from .text_utils import is_skip, normalize, validate_date, validate_money, validate_text

logger = logging.getLogger("hrbot.dialog")

# Эти поля добавляются автоматически к полям любой темы: заявитель — в начале, комментарий — в конце.
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
    """Полный список полей темы: заявитель -> поля темы -> комментарий."""
    return [*_LEADING_FIELDS, *intent.fields, *_TRAILING_FIELDS]


class Bot:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.sessions: dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        """Возвращает сессию по id, создавая новую при первом обращении."""
        session = self.sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self.sessions[session_id] = session
        return session

    def reply(self, session_id: str, message: str) -> dict:
        """Обрабатывает одно сообщение пользователя согласно текущей стадии диалога."""
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

        session.stage = "new"
        return self._handle_new(session, message, lower)

    # ------------------------------------------------------------------ #
    # Обработчики стадий
    # ------------------------------------------------------------------ #
    def _handle_new(self, session: Session, message: str, lower: str) -> dict:
        """Стадия «new»: классифицирует свободный запрос и решает, что делать дальше."""
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
        """Стадия «disambiguate»: ждёт выбора между предложенными темами."""
        offered = list(session.candidates)
        for key in session.candidates:
            intent_name_norm = normalize(self.classifier.intents[key].name)
            if intent_name_norm in lower or fuzz.WRatio(lower, intent_name_norm) >= 70:
                session.intent = key
                session.stage = "choose_action"
                session.candidates = []
                intent = self.classifier.intents[key]
                logger.info(
                    "route: disambiguation resolved -> intent=%s recipient=%s (offered %s)",
                    key, intent.recipient, offered,
                )
                return {
                    "text": (
                        f"Хорошо, «{intent.name}». "
                        "Вы хотите получить общую информацию или оформить обращение в профильную службу?"
                    ),
                    "state": "choose_action",
                    "intent": key,
                    "options": ["Получить информацию", "Оформить обращение"],
                }
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
        """Стадия «choose_action»: получить информацию или оформить обращение."""
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
        """Стадия «collecting»: последовательно собирает и валидирует поля темы."""
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
        """Стадия «confirm»: подтверждение и фактическая отправка письма."""
        if lower in _YES_WORDS or any(w in lower for w in ("отправ", "подтвер")):
            intent = self.classifier.intents[session.intent]
            logger.info(
                "send: session=%s intent=%s recipient=%s (about to dispatch)",
                session.session_id, intent.key, intent.recipient,
            )
            ok, info = send_email(intent, session.email_body or self.build_email(session))
            logger.info(
                "send: session=%s intent=%s recipient=%s ok=%s info=%s",
                session.session_id, intent.key, intent.recipient, ok, info,
            )
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
    # Вспомогательные методы
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_field(field_: Field, message: str) -> str | None:
        """Валидирует ответ пользователя по типу поля (text/date/money)."""
        if field_.kind == "date":
            return validate_date(message)
        if field_.kind == "money":
            return validate_money(message)
        return validate_text(message)

    @staticmethod
    def _retry_prompt(field_: Field) -> str:
        """Формирует текст повторного вопроса при неверном формате ответа."""
        if field_.kind == "date":
            return "Дата не распознана. Укажите её в формате ДД.ММ.ГГГГ, например 31.08.2026."
        if field_.kind == "money":
            return "Не вижу суммы в ответе. Укажите число, например: 45000 или 45 000 руб."
        return field_.label

    def _reset(self, session: Session) -> None:
        """Сбрасывает сессию к началу диалога."""
        session.intent = None
        session.stage = "new"
        session.field_cursor = 0
        session.values = {}
        session.candidates = []
        session.email_body = None

    def _sample_examples(self) -> str:
        """Возвращает пару примеров вопросов для подсказки при непонятном запросе."""
        pool = [ex for intent in self.classifier.intents.values() for ex in intent.examples[:1]]
        picks = pool[:2] if len(pool) >= 2 else pool
        return " / ".join(f"«{p}»" for p in picks) or "«как оформить увольнение сотрудника?»"

    def build_summary(self, session: Session) -> str:
        """Формирует текст сводки для подтверждения перед отправкой."""
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
        """Формирует текст письма из собранных данных сессии."""
        intent = self.classifier.intents[session.intent]
        fields = build_fields(intent)
        lines = [f"Обращение — {intent.name}", ""]
        for f in fields:
            value = session.values.get(f.key) or "—"
            lines.append(f"{f.title}: {value}")
        lines.append("")
        lines.append(f"Маршрут: {intent.recipient}")
        return "\n".join(lines)
