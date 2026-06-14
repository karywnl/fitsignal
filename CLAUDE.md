# CLAUDE.md

Guidance for working in this repo. Read before making changes.

## What this is

A ranker for the **Redrob "Intelligent Candidate Discovery & Ranking" hackathon**.
Goal: from a 100,000-candidate pool (`candidates.jsonl`), output the **top 100**
best-fit candidates for one fixed job description (**Senior AI Engineer — Founding
Team**) as a CSV: `candidate_id,rank,score,reasoning`.

The grader is hidden. Composite = `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP +
0.05·P@10` → **top-10 quality dominates**.

## The one thing to understand

The dataset is **adversarial and anti-keyword by design**. We verified by profiling
all 100k:

- **Skills are noise.** Every skill appears ~12,000 times, uniformly sprayed across
  all candidates. Never use the `skills` array as evidence of fit.
- **Signal = `current_title` + `career_history` free-text descriptions + trajectory
  + `redrob_signals` (behavioral availability).**
- The pool is stacked: ~68k non-tech roles (keyword-stuffer hosts), ~22k generic
  software, and only ~1.1k genuine AI/ML — which itself contains sub-traps
  (AI-Research = no-production trap, Computer-Vision = no-NLP/IR trap, Junior-ML =
  wrong band).
- ~80 **honeypots** with impossible profiles must stay out of the top 100
  (>10% honeypots in top 100 = disqualification).

## Architecture (two phases)

1. `precompute.py` — **offline, slow, allowed to exceed 5 min.** Embeds each
   candidate's career text (NOT skills) with `all-MiniLM-L6-v2` and saves aligned
   `.npy` arrays + JD facet embeddings to `artifacts/`.
2. `rank.py` — **the timed step: ≤5 min, ≤16 GB, CPU-only, no network.** Loads the
   precomputed embeddings and does only matrix math + the interpretable feature
   scoring in `common.py`. Falls back to inline embedding if no cache covers the
   given candidates (e.g. the ≤100-candidate sandbox sample).

All scoring logic lives in `common.py`:
- `classify_role` — title → bucket + base prior (the decisive feature).
- `extract_features` — structured, interpretable signals + honeypot flags.
- `evidence_score` / `location_score` / `behavioral_multiplier` / `base_score` /
  `final_score` — the scoring stack. **Tuning weights live at module top
  (`W_ROLE`, `W_SEM`, `W_EVID`, `W_BAND`, `W_LOC`).**
- `make_reasoning` — builds each row's justification **only from facts in the
  profile** (no hallucination; cites a real concern; varies across ranks).

## Hard constraints (do not break)

- The ranking step (`rank.py`) must run on CPU, no network, no hosted-LLM calls,
  within 5 min / 16 GB. The expensive embedding is precompute-only.
- Submission CSV: exactly 100 rows, ranks 1–100 unique, score non-increasing,
  equal scores tie-broken by `candidate_id` ascending. Always re-validate with
  `tools/validate_submission.py` after changes.

## Commands

```bash
uv sync
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python tools/validate_submission.py submission.csv
python tools/explore.py --candidates ./candidates.jsonl   # re-profile the pool
```

Note: the provided data lives under `data/[PUB] India_runs_data_and_ai_challenge/`.
The literal `[PUB]` brackets break shell globbing — quote the path.

## When tuning

Reuse the existing `artifacts/` embeddings (don't re-embed — it's slow). Iterate
on weights/regexes in `common.py`, re-run `rank.py` (seconds), and check the
top-100 bucket distribution + honeypot count printed at the end.
