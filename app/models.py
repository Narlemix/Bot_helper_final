"""Data structures shared across the classifier, the dialog engine, and the mailer.

Nothing here talks to FastAPI, SMTP, or the FAQ file directly — this module is just
the shape of the data, so it can be imported by every other module without pulling
in unrelated dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    """One question the bot needs answered before it can submit a request.

    `key` is the dictionary key the answer is stored under. `label` is the full
    question shown in chat while collecting the answer. `title` is a short
    2-4 word name for the same field, used in the confirmation summary and in
    the outgoing email (e.g. label="Укажите планируемую дату увольнения, например
    31.08.2026." vs title="Дата увольнения") — kept separate from `label` instead
    of derived from it, because truncating a full sentence into a short label
    reliably produces something awkward or wrong. `kind` selects the validator
    in `text_utils` (`"text"`, `"date"`, or `"money"`). `required=False` fields
    accept skip-words ("нет", "-", etc.) and show as "—" when skipped.
    """

    key: str
    label: str
    title: str
    kind: str = "text"  # "text" | "date" | "money"
    required: bool = True


@dataclass
class Intent:
    """One request category the bot can classify a message into and act on.

    `keywords`/`examples` come straight from the routing table and are what the
    classifier matches free-form text against. `fields` are the *category-specific*
    questions only — the dialog engine automatically prepends a "your full name"
    field and appends an optional "comment" field to every intent, so callers
    don't need to repeat those in every entry of data/faq.json.
    """

    key: str
    name: str
    recipient: str
    keywords: list[str]
    examples: list[str]
    safe_answer: str
    fields: list[Field] = field(default_factory=list)
    # Populated by the classifier at load time (keywords + examples combined) —
    # not read from faq.json directly.
    references: list[str] = field(default_factory=list)


@dataclass
class Session:
    """In-memory conversation state for one chat session (one browser tab / user).

    `stage` drives the dialog engine's state machine:
      "new"            -> waiting for a free-form message to classify
      "disambiguate"   -> bot asked "did you mean X or Y", waiting for a pick
      "choose_action"  -> bot asked "info or submit a request", waiting for a pick
      "collecting"     -> walking through `intent.fields` (see `field_cursor`)
      "confirm"        -> summary shown, waiting for yes/no
    `field_cursor` indexes into the *effective* field list (full_name + intent
    fields + comment) for the intent currently being collected.
    `values` holds every collected answer, keyed by `Field.key`.
    """

    session_id: str
    intent: str | None = None
    stage: str = "new"
    field_cursor: int = 0
    values: dict[str, str] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    email_body: str | None = None
