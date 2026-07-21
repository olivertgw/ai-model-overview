#!/usr/bin/env python3
"""
Benchmark Updater für benchmarks.json — voll dynamisch
=======================================================
1. Lädt die in Langdock nutzbaren Modelle von https://langdock.com/de/models
2. Lädt alle sechs Benchmark-Leaderboards über api_client.py (ZeroEval API)
3. Matcht Modelle dynamisch über normalisierte Namen (keine gepflegte Map)
4. Aktualisiert ausschließlich `models[].scores`, `meta.updatedAt` und
   `meta.updatedLabel`. Das `benchmarks`-Array bleibt unangetastet.

Regeln:
  - Nur überschreiben, wenn die API einen Wert liefert; sonst bleibt der
    bisherige Wert (auch null) unverändert.
  - Bei mehreren API-Varianten desselben Modells gewinnt der höchste Wert.
  - Scores werden auf die Skala 0–100 normalisiert (eine Nachkommastelle).
  - Neue Langdock-Modelle werden nur ergänzt, wenn für mindestens vier der
    sechs Benchmarks Werte vorliegen. Modelle werden nie entfernt.

Nutzung:
  python3 update_benchmarks.py            # benchmarks.json aktualisieren
  python3 update_benchmarks.py --dry-run  # nur Änderungen anzeigen
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_client

DATA_FILE = Path(__file__).resolve().parent.parent / "benchmarks.json"
LANGDOCK_URL = "https://langdock.com/de/models"
MIN_BENCHMARKS_FOR_NEW_MODEL = 4

# Benchmark-ID in benchmarks.json -> Benchmark-ID der ZeroEval API
BENCHMARK_MAP: dict[str, str] = {
    "gpqa-diamond": "gpqa",
    "mmlu-pro": "mmlu-pro",
    "mmmu-pro": "mmmu-pro",
    "ifeval": "ifeval",
    "swe-bench": "swe-bench-verified",
    "arc-agi-2": "arc-agi-v2",
}

# Tokens ohne Unterscheidungskraft beim Namensvergleich
NOISE_TOKENS = {"claude", "preview", "instruct", "latest", "high", "medium", "low", "thinking"}
# Varianten, die beim Präfix-Fallback nie zum Produktmodell gezählt werden
EXCLUDED_VARIANT_TOKENS = {"base"}

GERMAN_MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]

PROVIDER_COLORS = {
    "openai": "openai", "anthropic": "anthropic", "google": "google",
    "meta": "meta", "deepseek": "deepseek", "mistral": "mistral",
    "mistral ai": "mistral", "xai": "xai",
}


def norm(name: str) -> str:
    """Normalisiert einen Modellnamen für den Vergleich Langdock <-> API."""
    s = re.sub(r"\(([^)]*)\)", r" \1 ", name.lower())
    s = re.sub(r"[^a-z0-9.]+", " ", s)
    tokens = [
        t for t in s.split()
        if t not in NOISE_TOKENS and not re.fullmatch(r"\d{4}\.?\d{2}\.?\d{2}|\d{8}", t)
    ]
    return " ".join(tokens)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


def fetch_langdock_models(session) -> list[dict[str, str]]:
    """Liest Anbieter und Modellnamen aus der Langdock-Modellübersicht."""
    resp = session.get(LANGDOCK_URL, timeout=30)
    resp.raise_for_status()
    rows = re.findall(
        r'fs-provider="([^"]+)"[^>]*class="w-layout-grid models_row".*?'
        r'text-weight-medium">([^<]+)<',
        resp.text, re.S,
    )
    seen, models = set(), []
    for provider, name in rows:
        key = norm(name)
        if key and key not in seen:
            seen.add(key)
            models.append({"provider": provider.strip(), "name": name.strip()})
    return models


def normalize_score(score: float, max_score: float) -> float | int | None:
    """Rechnet einen API-Score auf die Skala 0–100 um (1 Nachkommastelle)."""
    value = score * 100 if max_score <= 1.0 else score
    value = round(value, 1)
    if not 0 <= value <= 100:
        return None
    return int(value) if value == int(value) else value


def fetch_api_scores() -> dict[str, dict[str, float | int]]:
    """Liefert {lokale Benchmark-ID: {norm(model_name): bester Score 0–100}}."""
    session = api_client.create_session()
    lookup: dict[str, dict[str, float | int]] = {}
    for local_id, ze_id in BENCHMARK_MAP.items():
        print(f"  Lade {local_id} ({ze_id}) ...", end=" ", flush=True)
        result = api_client.get_benchmark_leaderboard(session, ze_id)
        scores: dict[str, float | int] = defaultdict(lambda: -1)
        for entry in result.entries:
            value = normalize_score(entry.score, result.max_score)
            if value is None:
                continue
            key = norm(entry.model_name)
            scores[key] = max(scores[key], value)
        lookup[local_id] = dict(scores)
        print(f"{len(scores)} Modelle")
    return lookup


def match_scores(name: str, lookup: dict[str, dict[str, float | int]]) -> tuple[dict, bool]:
    """
    Sucht die Scores eines Modells über den normalisierten Namen.

    Erst exakter Vergleich; falls leer, Präfix-Fallback (API-Name beginnt mit
    dem Modellnamen, z.B. 'Mistral Large 3 675B Instruct' für 'Mistral Large 3'),
    wobei ausgeschlossene Varianten (z.B. Base-Modelle) ignoriert werden.
    Liefert ({benchmark_id: score}, fallback_verwendet).
    """
    key = norm(name)
    exact = {
        bench_id: scores[key]
        for bench_id, scores in lookup.items() if key in scores
    }
    if exact:
        return exact, False

    fuzzy: dict[str, float | int] = {}
    for bench_id, scores in lookup.items():
        candidates = [
            value for api_key, value in scores.items()
            if api_key.startswith(key + " ")
            and not EXCLUDED_VARIANT_TOKENS & set(api_key.split())
        ]
        if candidates:
            fuzzy[bench_id] = max(candidates)
    return fuzzy, bool(fuzzy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aktualisiert benchmarks.json mit Scores der llm-stats.com API.")
    parser.add_argument("--dry-run", action="store_true", help="nur Änderungen anzeigen, nichts schreiben")
    parser.add_argument("--file", type=Path, default=DATA_FILE, help=f"Pfad zur JSON-Datei (Standard: {DATA_FILE})")
    args = parser.parse_args()

    data = json.loads(args.file.read_text(encoding="utf-8"))
    session = api_client.create_session()

    print("Lade Langdock-Modellliste ...", end=" ", flush=True)
    try:
        langdock_models = fetch_langdock_models(session)
        print(f"{len(langdock_models)} Modelle")
    except Exception as error:
        langdock_models = []
        print(f"fehlgeschlagen ({error}) – bestehende Modellliste bleibt maßgeblich")

    lookup = fetch_api_scores()

    changes: list[str] = []
    fuzzy_matches: list[str] = []
    unmatched: list[str] = []
    added: list[str] = []

    def apply_scores(model: dict, api_scores: dict) -> None:
        for bench_id in BENCHMARK_MAP:
            if bench_id not in api_scores:
                continue  # kein API-Wert -> bisherigen Wert behalten
            new_value = api_scores[bench_id]
            old_value = model["scores"].get(bench_id)
            if new_value != old_value:
                changes.append(f"{model['id']:26} {bench_id:14} {old_value} -> {new_value}")
                model["scores"][bench_id] = new_value

    # 1) Bestehende Modelle aktualisieren (Modelle werden nie entfernt)
    existing_keys = set()
    for model in data["models"]:
        existing_keys.add(norm(model["name"]))
        api_scores, used_fallback = match_scores(model["name"], lookup)
        if not api_scores:
            unmatched.append(model["id"])
            continue
        if used_fallback:
            fuzzy_matches.append(f"{model['id']} (Präfix-Match)")
        apply_scores(model, api_scores)

    # 2) Neue Langdock-Modelle ergänzen, wenn genug Benchmarks belegt sind
    for entry in langdock_models:
        if norm(entry["name"]) in existing_keys:
            continue
        api_scores, used_fallback = match_scores(entry["name"], lookup)
        if len(api_scores) < MIN_BENCHMARKS_FOR_NEW_MODEL:
            continue
        provider = entry["provider"]
        model = {
            "id": slugify(entry["name"]),
            "name": entry["name"],
            "company": provider,
            "initial": provider[0].upper(),
            "color": PROVIDER_COLORS.get(provider.lower(), slugify(provider)),
            "scores": {bench_id: None for bench_id in BENCHMARK_MAP},
            "description": f"Ein Modell von {provider}.",
        }
        if used_fallback:
            fuzzy_matches.append(f"{model['id']} (Präfix-Match)")
        apply_scores(model, api_scores)
        data["models"].append(model)
        added.append(model["id"])

    data["models"].sort(key=lambda m: m["id"])

    today = datetime.date.today()
    data["meta"]["updatedAt"] = today.isoformat()
    data["meta"]["updatedLabel"] = f"{GERMAN_MONTHS[today.month - 1]} {today.year}"

    print(f"\n{len(changes)} Score-Änderungen:")
    for line in changes:
        print(f"  {line}")
    if added:
        print(f"\nNeu ergänzte Modelle: {', '.join(added)}")
    if fuzzy_matches:
        print(f"\nÜber Präfix-Fallback gematcht (bitte prüfen): {', '.join(fuzzy_matches)}")
    if unmatched:
        print(f"\nOhne API-Treffer (Werte unverändert): {', '.join(unmatched)}")

    if args.dry_run:
        print("\nDry-Run: nichts geschrieben.")
        return

    args.file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{args.file} aktualisiert (updatedAt = {data['meta']['updatedAt']}).")


if __name__ == "__main__":
    main()
