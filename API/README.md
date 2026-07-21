# LLM Stats Benchmark Data Extractor

Extracts AI model leaderboard/benchmark data from [llm-stats.com](https://llm-stats.com/) via the underlying **ZeroEval API** (`api.zeroeval.com`).

## Benchmarks Covered

| Benchmark | API ID | Models | Leader |
|---|---|---|---|
| GPQA (Diamond) | `gpqa` | 232 | GPT-5.6 Sol (94.6%) |
| MMLU-Pro | `mmlu-pro` | 129 | Qwen3.7 Max (89.6%) |
| MMMU-Pro | `mmmu-pro` | 64 | Gemini 3.5 Flash (83.6%) |
| IFEval | `ifeval` | 65 | Qwen3.5-27B (95.0%) |
| SWE-Bench Verified | `swe-bench-verified` | 104 | Claude Fable 5 (95.0%) |
| ARC-AGI v2 | `arc-agi-v2` | 16 | GPT-5.5 (85.0%) |

*Data as of July 21, 2026*

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests
```

## Usage

### Run the script directly

```bash
python3 api_client.py
```

This will:
1. Fetch all 6 benchmark leaderboards
2. Print formatted tables (top 20 per benchmark)
3. Save complete data to `benchmark_results.json`

### Use as a library

```python
from api_client import (
    create_session,
    get_benchmark_leaderboard,
    fetch_all_target_benchmarks,
    list_all_benchmarks,
)

session = create_session()

# Fetch a single benchmark
gpqa = get_benchmark_leaderboard(session, "gpqa")
for entry in gpqa.entries[:5]:
    print(f"{entry.rank}. {entry.model_name} ({entry.organization}): {entry.score:.3f}")

# Fetch all target benchmarks at once
results = fetch_all_target_benchmarks(session)
for bm_id, result in results.items():
    print(f"{result.benchmark_name}: {result.total_models} models")

# Limit entries per benchmark
results = fetch_all_target_benchmarks(session, max_entries=10)

# List ALL available benchmarks (200+)
all_benchmarks = list_all_benchmarks(session)
for bm in all_benchmarks[:10]:
    print(f"{bm['benchmark_id']}: {bm['name']} ({bm['model_count']} models)")
```

### Fetch any benchmark

The `get_benchmark_leaderboard()` function works with any benchmark ID from llm-stats.com:

```python
# Other popular benchmarks
aime = get_benchmark_leaderboard(session, "aime-2025")
humaneval = get_benchmark_leaderboard(session, "humaneval")
math = get_benchmark_leaderboard(session, "math")
livecodebench = get_benchmark_leaderboard(session, "livecodebench")
mmlu = get_benchmark_leaderboard(session, "mmlu")
hle = get_benchmark_leaderboard(session, "humanity's-last-exam")
```

## API Details

### Discovered Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/leaderboard/benchmarks` | GET | List all benchmarks with metadata |
| `/leaderboard/benchmarks/{id}` | GET | Full leaderboard for a specific benchmark |
| `/leaderboard/models/full` | GET | Main LLM leaderboard with composite scores |
| `/leaderboard/models/list` | GET | List all tracked models |
| `/leaderboard/organizations` | GET | List all organizations |
| `/leaderboard/providers` | GET | List all API providers |
| `/v1/models/metrics` | GET | Live model metrics (speed, latency) |

**Base URL:** `https://api.zeroeval.com`

### Authentication

No authentication required. The API is public but expects CORS headers:
- `Origin: https://llm-stats.com`
- `Referer: https://llm-stats.com/`

### Pagination

The `/leaderboard/benchmarks/{id}` endpoint returns 20 entries per request. Use the `offset` query parameter to paginate:

```
GET /leaderboard/benchmarks/gpqa?offset=0    # entries 1-20
GET /leaderboard/benchmarks/gpqa?offset=20   # entries 21-40
GET /leaderboard/benchmarks/gpqa?offset=40   # entries 41-60
```

### Response Format

```json
{
  "benchmark_id": "gpqa",
  "benchmark_name": "GPQA",
  "benchmark_description": "A challenging dataset of 448 multiple-choice questions...",
  "max_score": 1.0,
  "categories": ["physics", "reasoning", "general", "biology", "chemistry"],
  "modality": "text",
  "total_models": 232,
  "entries": [
    {
      "rank": 1,
      "model_id": "gpt-5.6-sol",
      "model_name": "GPT-5.6 Sol",
      "organization_name": "OpenAI",
      "organization_id": "openai",
      "benchmark_score": 0.946,
      "normalized_score": 0.946,
      "verified": false,
      "self_reported": true,
      "provider_id": "openai",
      "input_cost_per_million": 5.0,
      "output_cost_per_million": 30.0,
      "speed_rps": null,
      "context_window": 1050000,
      "release_date": "2026-07-09",
      "announcement_date": "2026-07-09",
      "multimodal": true,
      "param_count": null,
      "is_new": false
    }
  ]
}
```

### Entry Fields

| Field | Type | Description |
|---|---|---|
| `rank` | int | Position on the leaderboard |
| `model_id` | string | Unique model identifier (slug) |
| `model_name` | string | Human-readable model name |
| `organization_name` | string | Company/org name |
| `benchmark_score` | float | Raw score (0.0 - max_score) |
| `verified` | bool | Whether score is independently verified |
| `self_reported` | bool | Whether score is self-reported by the org |
| `input_cost_per_million` | float/null | Input token price per 1M tokens (USD) |
| `output_cost_per_million` | float/null | Output token price per 1M tokens (USD) |
| `context_window` | int/null | Maximum context length in tokens |
| `param_count` | int/null | Model parameter count |
| `release_date` | string/null | Release date (YYYY-MM-DD) |

## Output Files

- `api_client.py` - Main Python script
- `benchmark_results.json` - Full extracted data (auto-generated on run)
- `benchmark_data.json` - Raw data captured from browser session

## Caveats

- Scores are mostly **self-reported** by model providers, not independently verified
- The GPQA benchmark on llm-stats.com corresponds to **GPQA Diamond** accuracy
- ARC-AGI v2 has relatively few models (16) compared to other benchmarks
- API has no rate limiting observed, but be respectful with request frequency
- No authentication or API key needed - this is a public read-only API
