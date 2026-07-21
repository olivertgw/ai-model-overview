#!/usr/bin/env python3
"""
LLM Stats Benchmark Data Extractor
===================================
Extracts leaderboard/benchmark data from https://llm-stats.com/ via
the underlying ZeroEval API (api.zeroeval.com).

Benchmarks covered:
  - GPQA (Diamond)
  - MMLU-Pro
  - MMMU-Pro
  - IFEval
  - SWE-Bench Verified
  - ARC-AGI v2

API discovered by reverse-engineering llm-stats.com network traffic.
No authentication required - the API is public with CORS for llm-stats.com.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api.zeroeval.com"
REFERER = "https://llm-stats.com/"
ORIGIN = "https://llm-stats.com"

# Target benchmarks (benchmark_id -> display name)
TARGET_BENCHMARKS: dict[str, str] = {
    "gpqa": "GPQA (Diamond)",
    "mmlu-pro": "MMLU-Pro",
    "mmmu-pro": "MMMU-Pro",
    "ifeval": "IFEval",
    "swe-bench-verified": "SWE-Bench Verified",
    "arc-agi-v2": "ARC-AGI v2",
}

# Page size returned by the API (fixed at 20)
API_PAGE_SIZE = 20


# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create a requests session mimicking browser headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        # Note: do NOT set Accept-Encoding manually - let requests handle it.
        # The browser sends 'br, zstd' which Python can't decode natively.
        "Origin": ORIGIN,
        "Referer": REFERER,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
    })
    return session


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    """A single model's score on a benchmark."""
    rank: int
    model_id: str
    model_name: str
    organization: str
    score: float
    verified: bool
    self_reported: bool
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    context_window: int | None
    param_count: int | None
    release_date: str | None


@dataclass
class BenchmarkResult:
    """Full benchmark leaderboard data."""
    benchmark_id: str
    benchmark_name: str
    description: str
    max_score: float
    categories: list[str]
    total_models: int
    entries: list[BenchmarkEntry]


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def list_all_benchmarks(session: requests.Session) -> list[dict[str, Any]]:
    """
    List all available benchmarks on llm-stats.com.

    Returns:
        List of benchmark metadata dicts with keys:
        benchmark_id, name, description, categories, modality, max_score, model_count
    """
    url = f"{BASE_URL}/leaderboard/benchmarks"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_benchmark_leaderboard(
    session: requests.Session,
    benchmark_id: str,
    max_entries: int | None = None,
) -> BenchmarkResult:
    """
    Fetch the full leaderboard for a specific benchmark.

    Args:
        session: requests.Session with appropriate headers
        benchmark_id: The benchmark ID (e.g. 'gpqa', 'mmlu-pro')
        max_entries: Optional limit on number of entries to fetch.
                     None = fetch all.

    Returns:
        BenchmarkResult with all model scores
    """
    all_entries: list[BenchmarkEntry] = []
    offset = 0
    meta: dict[str, Any] = {}

    while True:
        url = f"{BASE_URL}/leaderboard/benchmarks/{benchmark_id}"
        params = {"offset": offset} if offset > 0 else {}

        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Capture metadata on first request
        if offset == 0:
            meta = {
                "benchmark_id": data.get("benchmark_id", benchmark_id),
                "benchmark_name": data.get("benchmark_name", benchmark_id),
                "description": data.get("benchmark_description", ""),
                "max_score": data.get("max_score", 1.0),
                "categories": data.get("categories", []),
                "total_models": data.get("total_models", 0),
            }

        entries = data.get("entries", [])
        if not entries:
            break

        for e in entries:
            all_entries.append(BenchmarkEntry(
                rank=e["rank"],
                model_id=e["model_id"],
                model_name=e["model_name"],
                organization=e.get("organization_name", ""),
                score=e["benchmark_score"],
                verified=e.get("verified", False),
                self_reported=e.get("self_reported", False),
                input_cost_per_million=e.get("input_cost_per_million"),
                output_cost_per_million=e.get("output_cost_per_million"),
                context_window=e.get("context_window"),
                param_count=e.get("param_count"),
                release_date=e.get("release_date"),
            ))

        offset += len(entries)

        if max_entries and len(all_entries) >= max_entries:
            all_entries = all_entries[:max_entries]
            break

        if len(all_entries) >= meta.get("total_models", float("inf")):
            break

        # Small delay to be respectful
        time.sleep(0.1)

    return BenchmarkResult(
        benchmark_id=meta.get("benchmark_id", benchmark_id),
        benchmark_name=meta.get("benchmark_name", benchmark_id),
        description=meta.get("description", ""),
        max_score=meta.get("max_score", 1.0),
        categories=meta.get("categories", []),
        total_models=meta.get("total_models", 0),
        entries=all_entries,
    )


def get_full_model_leaderboard(
    session: requests.Session,
    canonicals_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch the full model leaderboard with composite scores.

    This endpoint returns the main LLM leaderboard with pre-computed
    scores for key benchmarks (gpqa_score, swe_bench_verified_score, etc.)
    plus pricing, speed, and context data.

    Args:
        session: requests.Session
        canonicals_only: If True, only return canonical model entries

    Returns:
        List of model dicts with scores and metadata
    """
    url = f"{BASE_URL}/leaderboard/models/full"
    params = {"justCanonicals": str(canonicals_only).lower()}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_model_metrics(session: requests.Session) -> list[dict[str, Any]]:
    """
    Fetch live model metrics (speed, latency, pricing).

    Returns:
        List of model metric dicts
    """
    url = f"{BASE_URL}/v1/models/metrics"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def fetch_all_target_benchmarks(
    session: requests.Session | None = None,
    max_entries: int | None = None,
) -> dict[str, BenchmarkResult]:
    """
    Fetch leaderboard data for all target benchmarks.

    Args:
        session: Optional pre-configured session
        max_entries: Optional limit per benchmark

    Returns:
        Dict mapping benchmark_id to BenchmarkResult
    """
    if session is None:
        session = create_session()

    results: dict[str, BenchmarkResult] = {}
    for bm_id, display_name in TARGET_BENCHMARKS.items():
        print(f"  Fetching {display_name} ({bm_id})...", end=" ", flush=True)
        result = get_benchmark_leaderboard(session, bm_id, max_entries)
        results[bm_id] = result
        print(f"{len(result.entries)} models")

    return results


def to_json(results: dict[str, BenchmarkResult]) -> str:
    """Serialize benchmark results to JSON."""
    data = {}
    for bm_id, result in results.items():
        data[bm_id] = {
            "benchmark_id": result.benchmark_id,
            "benchmark_name": result.benchmark_name,
            "description": result.description,
            "max_score": result.max_score,
            "categories": result.categories,
            "total_models": result.total_models,
            "entries": [asdict(e) for e in result.entries],
        }
    return json.dumps(data, indent=2, ensure_ascii=False)


def print_leaderboard_table(result: BenchmarkResult, top_n: int = 20) -> None:
    """Print a formatted leaderboard table for a benchmark."""
    entries = result.entries[:top_n]
    if not entries:
        print(f"  No entries for {result.benchmark_name}")
        return

    # Header
    print(f"\n{'='*80}")
    print(f"  {result.benchmark_name} Leaderboard")
    print(f"  {result.total_models} models | max score: {result.max_score}")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'Model':<35} {'Org':<20} {'Score':>8}")
    print(f"  {'-'*3}  {'-'*35} {'-'*20} {'-'*8}")

    for entry in entries:
        score_pct = f"{entry.score * 100:.1f}%" if result.max_score <= 1.0 else f"{entry.score:.1f}"
        org = entry.organization[:18] if entry.organization else "—"
        name = entry.model_name[:33] if entry.model_name else "—"
        print(f"  {entry.rank:>3}  {name:<35} {org:<20} {score_pct:>8}")

    if len(result.entries) > top_n:
        print(f"  ... and {len(result.entries) - top_n} more models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch and display benchmark data from llm-stats.com."""
    print("LLM Stats Benchmark Extractor")
    print("=" * 40)
    print()

    session = create_session()

    # Fetch all target benchmarks
    print("Fetching benchmark data...")
    results = fetch_all_target_benchmarks(session)
    print()

    # Display top 20 for each benchmark
    for bm_id, result in results.items():
        print_leaderboard_table(result, top_n=20)

    # Save full data to JSON
    output_path = "benchmark_results.json"
    json_data = to_json(results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_data)
    print(f"\nFull data saved to {output_path}")

    # Summary
    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)
    for bm_id, result in results.items():
        top = result.entries[0] if result.entries else None
        if top:
            score_str = f"{top.score * 100:.1f}%" if result.max_score <= 1.0 else f"{top.score:.1f}"
            print(f"  {result.benchmark_name:<25} Leader: {top.model_name} ({score_str})")
        else:
            print(f"  {result.benchmark_name:<25} No data")


if __name__ == "__main__":
    main()
