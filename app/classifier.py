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
import os
from pathlib import Path

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Field, Intent
from .text_utils import normalize

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
        """
        query_norm = normalize(text)
        if not query_norm:
            return None, 0.0, []

        combined = self._scores_per_intent(query_norm)
        ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked:
            return None, 0.0, []

        best_key, best_score = ranked[0]
        if best_score < NO_MATCH_THRESHOLD:
            return None, best_score, ranked

        if len(ranked) > 1:
            _, second_score = ranked[1]
            if best_score < AMBIGUITY_CEILING and (best_score - second_score) < AMBIGUITY_MARGIN:
                # Ambiguous: caller decides how to prompt using `ranked`.
                return None, best_score, ranked

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
