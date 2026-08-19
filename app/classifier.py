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

# Ниже этого порога бот считает, что не понял запрос.
NO_MATCH_THRESHOLD = 0.30

# Если разрыв между лучшим и вторым вариантом меньше этого значения — переспрашиваем.
AMBIGUITY_MARGIN = 0.06
AMBIGUITY_CEILING = 0.60

# Веса при объединении двух сигналов схожести.
COSINE_WEIGHT = 0.45
FUZZY_WEIGHT = 0.55

# Пары тем-«близнецов»: похожи почти дословно, различаются одним словом.
# Если в запросе нет ни одного отличительного слова — всегда переспрашиваем,
# какой вариант нужен, независимо от итоговых баллов.
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
    # Классификация
    # ------------------------------------------------------------------ #
    def _scores_per_intent(self, query_norm: str) -> dict[str, float]:
        """Считает совмещённый (cosine + fuzzy) балл для каждой темы."""
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

        # Короткие запросы дисконтируем: иначе одно общее слово ложно
        # набирает высокий балл почти у всех тем сразу.
        length_confidence = min(1.0, len(query_norm) / 14.0)
        best_fuzzy = {key: score * length_confidence for key, score in best_fuzzy.items()}

        combined: dict[str, float] = {}
        for key in self.intents:
            combined[key] = COSINE_WEIGHT * best_cosine.get(key, 0.0) + FUZZY_WEIGHT * best_fuzzy.get(key, 0.0)
        return combined

    def classify(self, text: str) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Определяет тему свободного запроса; логирует решение в hrbot.classifier."""
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
                continue
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
    # Подсказки при вводе
    # ------------------------------------------------------------------ #
    def suggest(self, query: str, limit: int = 6) -> list[str]:
        """Возвращает до `limit` подсказок для поля ввода, устойчиво к опечаткам."""
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
