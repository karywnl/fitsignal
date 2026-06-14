#!/usr/bin/env python3
"""Offline precompute step (allowed to exceed the 5-minute ranking budget).

Streams candidates.jsonl, builds embedding documents (career text — NOT skills),
embeds them with a CPU sentence-transformer, and saves aligned arrays plus the
JD facet embeddings. rank.py then loads these and only does fast matrix math.

Usage:
    python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import build_doc, JD_POSITIVE_FACETS, JD_NEGATIVE_FACETS
from embed import embed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    t0 = time.time()
    ids, docs = [], []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            ids.append(c["candidate_id"])
            docs.append(build_doc(c))
    print(f"Loaded {len(ids)} candidates in {time.time() - t0:.1f}s; embedding...")

    # Embed in chunks with visible, flushed ETA logging (the encode is the long
    # pole; the rest of the pipeline is seconds).
    t1 = time.time()
    chunks = []
    total = len(docs)
    step = max(args.batch_size * 8, 2000)
    for start in range(0, total, step):
        part = docs[start:start + step]
        chunks.append(embed_texts(part, batch_size=args.batch_size))
        done = start + len(part)
        elapsed = time.time() - t1
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(f"  embedded {done}/{total}  ({rate:.0f}/s, ETA {eta/60:.1f} min)",
              flush=True)
    emb = np.vstack(chunks).astype(np.float32)
    print(f"Embedded {emb.shape} in {time.time() - t1:.1f}s")

    jd_pos = embed_texts(JD_POSITIVE_FACETS)
    jd_neg = embed_texts(JD_NEGATIVE_FACETS)

    np.save(os.path.join(args.artifacts, "cand_emb.npy"), emb)
    np.save(os.path.join(args.artifacts, "cand_ids.npy"), np.array(ids))
    np.save(os.path.join(args.artifacts, "jd_pos.npy"), jd_pos)
    np.save(os.path.join(args.artifacts, "jd_neg.npy"), jd_neg)
    print(f"Saved artifacts to {args.artifacts} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
