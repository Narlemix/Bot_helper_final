"""Turns a free-form Russian message into a request category (an `Intent`).

Two complementary signals are combined, because either one alone is fragile on
short, informal, occasionally misspelled input like real chat messages:

- **TF-IDF over character 3–5 grams + cosine similarity.** Robust to typos and
  word endings/case (Russian is heavily inflected) because it compares
  overlapping substrings rather than whole words — "вахтовика" and "вахтовик"
  share almost all their trigrams even though they're different tokens.
- **rapidfuzz.WRatio**, a word-level fuzzy ratio. Robust to word order and to
  a query being a short fragment of a longer reference phrase (partial-ratio
  component), which char-ngrams alone under-score.

The same machinery (`self.suggestion_pool`, `normalize`) also powers the
`/api/suggest` typeahead endpoint, so the "did you mean" behavior in chat and
the live dropdown while typing are always in sync with the same phrase list.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Field, Intent
from .text_utils import normalize

logger = logging.getLogger("hrbot.classifier")

# Below this combined score, the bot admits it didn't understand rather than
# guessing at a category.
NO_MATCH_THRESHOLD = 0.30

# If the best and second-best intents are within this margin of each other
# (and the best isn't confident on its own), ask a disambiguating question
# instead of silently picking one. Keeping the ceiling relatively low (rather
# than e.g. 0.85) matters: with 17 categories sharing a lot of common HR/admin
# vocabulary, a "close second" is normal even when the top pick is clearly
# right — disambiguation should kick in for genuinely low-confidence calls,
# not every time two categories happen to share a few words.
AMBIGUITY_MARGIN = 0.06
AMBIGUITY_CEILING = 0.60

# Weights for combining the two similarity signals into one score.
COSINE_WEIGHT = 0.45
FUZZY_WEIGHT = 0.55

# Some intent pairs are *near-duplicates by design* — e.g. "увольнение офиса"
# and "увольнение вахтовика" share almost identical wording and differ mainly
# in whether the employee is a вахтовик. On a query that doesn't lean toward
# either side ("уволить сотрудника"), both score high AND close together,
# which normally would trigger the ambiguity check above — but
# AMBIGUITY_CEILING intentionally lets high-confidence top picks through
# without a second look (see the comment above it), and both members of a
# pair like this *are* high-confidence on their own, just for the wrong
# reason. So: for each listed group, if the query contains none of either
# side's disambiguating words, force a clarifying question regardless of how
# high the scores are — the ceiling doesn't get a vote for these specific
# pairs. If the query *does* contain one side's word (e.g. "офиса" or
# "вахтовика"), it already disambiguates itself and normal scoring proceeds.
#
# (group of intent keys, {intent_key: [substrings that mean "this one, no
#  need to ask"] for each member that has such a word})
CONFUSABLE_GROUPS: list[tuple[frozenset[str], dict[str, list[str]]]] = [
    (
        frozenset({"admin_dismissal", "vahta_dismissal"}),
        {
            "vahta_dismissal": ["вахт"],
            "admin_dismissal": ["офис", "администр"],
        },
    ),
]


class Classifier:
    def __init__(self, faq_path: Path):
        raw = json.loads(faq_path.read_text(encoding="utf-8"))

        self.intents: dict[str, Intent] = {}
        docs: list[str] = []
        self.doc_intent_keys: list[str] = []
        suggestion_pool: list[str] = []
        seen_suggestions: set[str] = set()

        for key, item in raw.items():
            # Per-intent recipient override, e.g. RECIPIENT_EMAIL_ADMIN_CLAIMS=...
            # Falls back to the address defined in faq.json. There is deliberately
            # no *global* RECIPIENT_EMAIL override anymore: with 17 categories
            # routing to different inboxes, a single global override would send
            # everything to one address and silently break routing.
            env_override = os.getenv(f"RECIPIENT_EMAIL_{key.upper()}")
            fields = [
                Field(
                    key=f["key"],
                    label=f["label"],
                    title=f["title"],
                    kind=f.get("kind", "text"),
                    required=f.get("required", True),
                )
                for f in item.get("fields", [])
            ]
            intent = Intent(
                key=key,
                name=item["name"],
                recipient=env_override or item["recipient"],
                keywords=item["keywords"],
                examples=item["example_questions"],
                safe_answer=item["safe_answer"],
                fields=fields,
            )
            intent.references = [*intent.keywords, *intent.examples]
            self.intents[key] = intent

            for ref in intent.references:
                docs.append(normalize(ref))
                self.doc_intent_keys.append(key)
                if ref not in seen_suggestions:
                    seen_suggestions.add(ref)
                    suggestion_pool.append(ref)

        if not docs:
            raise ValueError(f"No intents/keywords loaded from {faq_path}")

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs)
        self.suggestion_pool = suggestion_pool

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _scores_per_intent(self, query_norm: str) -> dict[str, float]:
        """Combined cosine+fuzzy score for every intent, keyed by intent key."""
        query_vec = self.vectorizer.transform([query_norm])
        cosine_scores = cosine_similarity(query_vec, self.matrix)[0]

        best_cosine: dict[str, float] = {}
        for doc_idx, key in enumerate(self.doc_intent_keys):
            score = float(cosine_scores[doc_idx])
            if score > best_cosine.get(key, -1.0):
                best_cosine[key] = score

        best_fuzzy: dict[str, float] = {}
        for key, intent in self.intents.items():
            best = 0.0
            for ref in intent.references:
                ratio = fuzz.WRatio(query_norm, normalize(ref)) / 100.0
                if ratio > best:
                    best = ratio
            best_fuzzy[key] = best

        # WRatio leans on partial-ratio (substring containment), which inflates
        # scores for very short queries: a single common word like "документы"
        # sits inside almost every reference phrase and would otherwise score
        # ~0.75+ against nearly all 17 intents simultaneously. Discount the
        # fuzzy signal for short queries so a bare word doesn't out-rank the
        # ambiguity ceiling and get committed to a single category by accident —
        # it should instead fall through to "ask for clarification".
        length_confidence = min(1.0, len(query_norm) / 14.0)
        best_fuzzy = {key: score * length_confidence for key, score in best_fuzzy.items()}

        combined: dict[str, float] = {}
        for key in self.intents:
            combined[key] = COSINE_WEIGHT * best_cosine.get(key, 0.0) + FUZZY_WEIGHT * best_fuzzy.get(key, 0.0)
        return combined

    def classify(self, text: str) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Classify a free-form message.

        Returns `(best_key_or_None, best_score, ranked_alternatives)`.
        `ranked_alternatives` is every intent sorted by score, descending —
        callers use the top 2-3 to build a "did you mean X or Y?" prompt when
        the match is ambiguous.

        Every outcome is logged at INFO under "hrbot.classifier" — including
        the recipient email address the request would route to — specifically
        so routing decisions can be watched live in the hosting provider's log
        viewer (Railway "Logs" tab, `docker compose logs -f`, etc.) without
        needing access to the code or a debugger. See README.md → "Логи
        маршрутизации".
        """
        query_norm = normalize(text)
        if not query_norm:
            return None, 0.0, []

        combined = self._scores_per_intent(query_norm)
        ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked:
            return None, 0.0, []

        top3 = ", ".join(f"{key}={score:.2f}" for key, score in ranked[:3])
        best_key, best_score = ranked[0]

        if best_score < NO_MATCH_THRESHOLD:
            logger.info("route: query=%r -> NO_MATCH (top3: %s)", text.strip(), top3)
            return None, best_score, ranked

        for group, disambiguators in CONFUSABLE_GROUPS:
            if best_key not in group:
                continue
            if any(word in query_norm for words in disambiguators.values() for word in words):
                continue  # the query already leans toward one side — trust normal scoring
            # Neither side's word is present, so no matter how skewed the score
            # happens to be (due to incidental wording overlap), we genuinely
            # can't tell which one the person means — ask, don't guess.
            group_ranked = [(key, score) for key, score in ranked if key in group]
            g_key, g_score = group_ranked[0]
            g2_key, g2_score = group_ranked[1]
            logger.info(
                "route: query=%r -> AMBIGUOUS (confusable pair, no disambiguating word) between %s(%.2f) and %s(%.2f) (top3: %s)",
                text.strip(), g_key, g_score, g2_key, g2_score, top3,
            )
            return None, best_score, ranked

        if len(ranked) > 1:
            second_key, second_score = ranked[1]
            if best_score < AMBIGUITY_CEILING and (best_score - second_score) < AMBIGUITY_MARGIN:
                logger.info(
                    "route: query=%r -> AMBIGUOUS between %s(%.2f) and %s(%.2f) (top3: %s)",
                    text.strip(), best_key, best_score, second_key, second_score, top3,
                )
                return None, best_score, ranked

        intent = self.intents[best_key]
        logger.info(
            "route: query=%r -> intent=%s recipient=%s score=%.2f (top3: %s)",
            text.strip(), best_key, intent.recipient, best_score, top3,
        )
        return best_key, best_score, ranked

    # ------------------------------------------------------------------ #
    # Typeahead
    # ------------------------------------------------------------------ #
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
