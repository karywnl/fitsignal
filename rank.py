#!/usr/bin/env python3
"""Ranking step — produces the top-100 submission CSV.

Runs within the contest budget (<=5 min, <=16GB, CPU, no network): it loads the
precomputed embeddings from precompute.py and only does fast matrix math + the
interpretable feature scoring in common.py.

If no precomputed cache covers the candidates (e.g. the small sandbox sample),
it embeds them inline — fine for <=100 candidates.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from common import (
    build_doc, extract_features, final_score, make_reasoning,
    JD_POSITIVE_FACETS, JD_NEGATIVE_FACETS,
)


def load_candidates(path):
    feats, docs, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            feats.append(extract_features(c))
            docs.append(build_doc(c))
            ids.append(c["candidate_id"])
    return ids, docs, feats


def get_embeddings(ids, docs, artifacts):
    """Return (cand_emb aligned to ids, jd_pos, jd_neg). Use the precomputed
    cache when it covers all ids; otherwise embed inline (sandbox fallback)."""
    cand_path = os.path.join(artifacts, "cand_emb.npy")
    ids_path = os.path.join(artifacts, "cand_ids.npy")
    if os.path.exists(cand_path) and os.path.exists(ids_path):
        cache_ids = np.load(ids_path, allow_pickle=True)
        index = {cid: i for i, cid in enumerate(cache_ids.tolist())}
        if all(cid in index for cid in ids):
            cache_emb = np.load(cand_path)
            rows = np.array([index[cid] for cid in ids])
            cand_emb = cache_emb[rows]
            jd_pos = np.load(os.path.join(artifacts, "jd_pos.npy"))
            jd_neg = np.load(os.path.join(artifacts, "jd_neg.npy"))
            print(f"Using precomputed embeddings for {len(ids)} candidates.")
            return cand_emb, jd_pos, jd_neg

    print(f"No full cache; embedding {len(ids)} candidates inline...")
    from embed import embed_texts
    cand_emb = embed_texts(docs, show_progress=len(ids) > 1000)
    jd_pos = embed_texts(JD_POSITIVE_FACETS)
    jd_neg = embed_texts(JD_NEGATIVE_FACETS)
    return cand_emb, jd_pos, jd_neg


def semantic_scores(cand_emb, jd_pos, jd_neg):
    """Per-candidate semantic JD match: mean of the top-2 positive facet cosines
    minus a fraction of the strongest negative (anti-pattern) facet cosine.
    Embeddings are L2-normalized, so dot product == cosine."""
    pos = cand_emb @ jd_pos.T              # (N, P)
    neg = cand_emb @ jd_neg.T              # (N, Q)
    top2 = np.sort(pos, axis=1)[:, -2:]
    pos_score = top2.mean(axis=1)
    neg_score = neg.max(axis=1)
    return pos_score - 0.6 * neg_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    ids, docs, feats = load_candidates(args.candidates)
    print(f"Parsed {len(ids)} candidates in {time.time() - t0:.1f}s")

    cand_emb, jd_pos, jd_neg = get_embeddings(ids, docs, args.artifacts)
    sem = semantic_scores(cand_emb, jd_pos, jd_neg)

    # Percentile-rank normalization: spreads semantic scores uniformly over [0,1]
    # with no clustering at the ceiling, so the elite tier stays differentiated
    # (avoids many candidates tying at the maximum score).
    order = sem.argsort()
    ranks = np.empty(len(sem), dtype=np.float64)
    ranks[order] = np.arange(len(sem))
    sem_n = ranks / max(len(sem) - 1, 1)

    scored = []
    for i, f in enumerate(feats):
        s = final_score(f, float(sem_n[i]))
        scored.append((s, f, float(sem_n[i])))

    # Rank: score desc, then candidate_id asc (deterministic tie-break).
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top = scored[: args.topk]

    # Normalize the top scores to a clean (0,1] range for presentation; this is
    # monotonic so it does not change the ranking.
    smax = max((s for s, _, _ in top), default=1.0) or 1.0
    rows = []
    for s, f, sn in top:
        rows.append((round(s / smax, 6), f["candidate_id"], f, sn))
    # Re-sort so equal rounded scores tie-break by id ascending (exactly what
    # validate_submission.py checks).
    rows.sort(key=lambda r: (-r[0], r[1]))

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, f, sn) in enumerate(rows, start=1):
            w.writerow([cid, rank, f"{score:.6f}", make_reasoning(f, sn)])

    print(f"Wrote {len(rows)} rows -> {args.out} in {time.time() - t0:.1f}s total")

    # --- Diagnostics: distribution and trap checks over the top-100 ---
    from collections import Counter
    buckets = Counter(f["bucket"] for _, _, f, _ in rows)
    honeypots = sum(1 for _, _, f, _ in rows if f["honeypot"])
    scores_only = [s for s, _, _, _ in rows]
    print(f"  distinct scores: {len(set(scores_only))}/{len(rows)}  "
          f"range [{min(scores_only):.4f}, {max(scores_only):.4f}]")
    print(f"  honeypots in top-100: {honeypots}  (must be 0; >10 disqualifies)")
    print(f"  bucket mix: {dict(buckets.most_common())}")
    print("  top 10:")
    for rank, (score, cid, f, sn) in enumerate(rows[:10], start=1):
        print(f"  {rank:3d} {cid} {score:.4f} {f['bucket']:12s} "
              f"yoe={f['yoe']:.1f} ir={int(f['ev_ir'])} prod={int(f['ev_prod'])} "
              f"eval={int(f['ev_eval'])} india={int(f['in_india'])} sem={sn:.3f}")


if __name__ == "__main__":
    main()
