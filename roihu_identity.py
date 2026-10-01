#!/usr/bin/env python3
"""Conservative EP24 codebook identity resolution and human seed recycling.

This module is intentionally public and data-agnostic. Real EP24 labels stay in
LaclauGPT-Private; tests use invented fixtures only.

Resolution policy:
1. exact canonical/local/English label or alias;
2. normalized exact match using roihu_codebooks.identity_key;
3. ambiguity => abstain;
4. optional fuzzy similarity => candidate(s) only, never accepted automatically.

The same resolver is used for persons/entities, themes, and sentiment targets so
those annotations share one identity universe where the codebook permits it.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from roihu_codebooks import CodebookEntry, identity_key

ENTITY_KINDS = ("person", "actor", "entity", "organization", "party", "institution")
THEME_KINDS = ("topic", "theme")
SENTIMENT_KINDS = ENTITY_KINDS + THEME_KINDS


def _scope(entries: Iterable[CodebookEntry], *, country: str, kinds: Sequence[str]) -> list[CodebookEntry]:
    wanted = {kind.casefold() for kind in kinds}
    code = country.upper()
    return [
        entry for entry in entries
        if entry.kind.casefold() in wanted and entry.country in {"", "COMMON", code}
    ]


def _record(entry: CodebookEntry, *, observed: str, method: str, score: float = 1.0) -> dict[str, Any]:
    return {
        "observed": observed,
        "decision": "EXISTING",
        "entry_id": entry.entry_id,
        "kind": entry.kind,
        "canonical_label": entry.label,
        "english_label": entry.english_label,
        "country": entry.country,
        "match_method": method,
        "score": round(float(score), 6),
        "review_state": entry.review_state,
        "origin": entry.origin,
    }


def resolve_surface(
    observed: Any,
    entries: Iterable[CodebookEntry],
    *,
    country: str,
    kinds: Sequence[str],
    fuzzy_threshold: float = 0.86,
    fuzzy_limit: int = 5,
) -> dict[str, Any]:
    """Resolve one surface form conservatively.

    Fuzzy results are explicitly returned as CANDIDATE and therefore cannot
    silently canonize a politician, party, organization, theme, or sentiment
    target.
    """
    raw = str(observed or "").strip()
    if not raw:
        return {"observed": raw, "decision": "EMPTY", "candidates": []}

    scoped = _scope(entries, country=country, kinds=kinds)
    key = identity_key(raw)
    exact: list[tuple[CodebookEntry, str]] = []
    for entry in scoped:
        if identity_key(entry.label) == key:
            exact.append((entry, "exact_canonical"))
            continue
        if entry.english_label and identity_key(entry.english_label) == key:
            exact.append((entry, "exact_english_label"))
            continue
        if any(identity_key(alias) == key for alias in entry.aliases):
            exact.append((entry, "exact_alias"))

    unique = {entry.entry_id: (entry, method) for entry, method in exact}
    if len(unique) == 1:
        entry, method = next(iter(unique.values()))
        return _record(entry, observed=raw, method=method)
    if len(unique) > 1:
        return {
            "observed": raw,
            "decision": "AMBIGUOUS",
            "match_method": "normalized_exact_collision",
            "candidates": [
                _record(entry, observed=raw, method=method)
                for entry, method in sorted(unique.values(), key=lambda pair: pair[0].entry_id)
            ],
        }

    ranked: list[tuple[float, CodebookEntry, str]] = []
    for entry in scoped:
        best_score = 0.0
        best_form = ""
        for form in entry.forms:
            form_key = identity_key(form)
            if not form_key:
                continue
            score = SequenceMatcher(None, key, form_key).ratio()
            if score > best_score:
                best_score, best_form = score, form
        if best_score >= fuzzy_threshold:
            ranked.append((best_score, entry, best_form))
    ranked.sort(key=lambda item: (-item[0], item[1].entry_id))

    if ranked:
        return {
            "observed": raw,
            "decision": "CANDIDATE",
            "match_method": "fuzzy_candidate_only",
            "candidates": [
                {
                    **_record(entry, observed=raw, method="fuzzy_candidate", score=score),
                    "matched_form": form,
                    "decision": "CANDIDATE",
                }
                for score, entry, form in ranked[:max(1, fuzzy_limit)]
            ],
        }
    return {
        "observed": raw,
        "decision": "NEW",
        "match_method": "no_match",
        "candidates": [],
    }


def resolve_many(
    values: Iterable[str],
    entries: Iterable[CodebookEntry],
    *,
    country: str,
    kinds: Sequence[str],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        key = identity_key(raw)
        if key in seen:
            continue
        seen.add(key)
        results.append(resolve_surface(raw, entries, country=country, kinds=kinds))
    return results


def canonical_labels(results: Iterable[dict[str, Any]]) -> list[str]:
    """Return accepted canonical labels only. Candidates/ambiguities are excluded."""
    labels: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.get("decision") != "EXISTING":
            continue
        label = str(result.get("canonical_label") or "").strip()
        key = identity_key(label)
        if label and key not in seen:
            seen.add(key)
            labels.append(label)
    return labels


def seed_context_lines(kind: str, results: Iterable[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for result in results:
        decision = result.get("decision")
        if decision == "EXISTING":
            local = result.get("canonical_label") or result.get("observed")
            english = result.get("english_label")
            label = f"{local} / {english}" if english and english != local else str(local)
            lines.append(f"- {kind}: {label} [human-informed seed; verify from current item]")
        elif decision in {"AMBIGUOUS", "CANDIDATE", "NEW"}:
            lines.append(
                f"- {kind}: {result.get('observed', '')} "
                f"[{str(decision).lower()}; contextual hint only, do not treat as evidence]"
            )
    return lines
