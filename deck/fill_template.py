#!/usr/bin/env python3
"""Fill FitSignal's content onto the official India.Runs idea-submission template,
keeping the template's exact design (branding, headers, questions) and adding our
answers plus drawn diagrams for the Workflow and Architecture slides.

Run:  uv run --with pymupdf python deck/fill_template.py
Out:  deck/fitsignal_deck.pdf
"""
from __future__ import annotations

import os
import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "docs", "Idea Submission Template _ Redrob.pdf")
OUT = os.path.join(HERE, "fitsignal_deck.pdf")

DARK = (0.16, 0.17, 0.21)
GREY = (0.33, 0.34, 0.40)
PURPLE = (0.49, 0.30, 0.78)
BLUE = (0.20, 0.28, 0.78)
BOXFILL = (0.96, 0.95, 0.995)
BOXEDGE = (0.60, 0.52, 0.84)
ACCENTFILL = (0.92, 0.90, 0.99)

F = "helv"
FB = "hebo"


def wrap(text, size, maxw, font=F):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if fitz.get_text_length(t, fontname=font, fontsize=size) <= maxw:
            cur = t
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def bullets(page, x, y, items, width, size=9.3, lh=12.2, gap=5):
    """Draw purple-dot bullets with wrapped dark text. Returns the new y."""
    for it in items:
        page.draw_circle(fitz.Point(x + 2, y - 2.6), 1.7, color=PURPLE, fill=PURPLE)
        for i, ln in enumerate(wrap(it, size, width - 14)):
            page.insert_text(fitz.Point(x + 12, y), ln, fontname=F, fontsize=size,
                             color=DARK)
            y += lh
        y += gap
    return y


def centered(page, rect, lines, size, color, font=F):
    th = len(lines) * size * 1.18
    yy = rect.y0 + (rect.height - th) / 2 + size * 0.82
    for ln in lines:
        w = fitz.get_text_length(ln, fontname=font, fontsize=size)
        page.insert_text(fitz.Point(rect.x0 + (rect.width - w) / 2, yy), ln,
                         fontname=font, fontsize=size, color=color)
        yy += size * 1.18


def box(page, rect, label, fill=BOXFILL, edge=BOXEDGE, tcol=DARK, size=8.4,
        font=F):
    page.draw_rect(rect, color=edge, fill=fill, width=1)
    centered(page, rect, wrap(label, size, rect.width - 8, font), size, tcol, font)


def arrow(page, p1, p2, color=PURPLE, width=1.5):
    p1, p2 = fitz.Point(*p1), fitz.Point(*p2)
    page.draw_line(p1, p2, color=color, width=width)
    import math
    ang = math.atan2(p2.y - p1.y, p2.x - p1.x)
    for da in (math.radians(150), math.radians(-150)):
        page.draw_line(p2, fitz.Point(p2.x + 6 * math.cos(ang + da),
                                      p2.y + 6 * math.sin(ang + da)),
                       color=color, width=width)


# --------------------------------------------------------------------------
ANSWERS = {
    1: [  # Solution Overview
        "Proposed solution: a hybrid ranker with three lenses. Semantic embeddings "
        "read each candidate's career story, interpretable recruiter rules judge "
        "the real role and production evidence, and behavioral signals plus "
        "trap/honeypot guards refine the result.",
        "What sets it apart: we proved the skills list is noise (every skill "
        "appears about 12,000 times, sprayed across all profiles), so we ignore it "
        "and read the career narrative instead. We also model the JD's "
        "anti-patterns (research-only, computer vision, consulting, title-chasers) "
        "as negative signals, which keyword matchers cannot do.",
    ],
    2: [  # JD Understanding & Candidate Evaluation
        "Key requirements: production embeddings and retrieval, vector DB or hybrid "
        "search, ranking evaluation (NDCG, MRR, MAP), strong Python, 5 to 9 years "
        "at product (not services) companies, Pune or Noida, active on the "
        "platform. Disqualifiers: pure research, recent LangChain-only work, "
        "title-chasers, consulting-only careers, CV/speech without NLP or IR.",
        "Signals that matter most: current and past job title, career-history free "
        "text showing real production retrieval and ranking work, product vs "
        "services trajectory and tenure, and behavioral availability. Beyond "
        "keywords we use semantic match of the career story plus role "
        "classification and negative-facet penalties.",
    ],
    3: [  # Ranking Methodology
        "Retrieve, score, rank: embed each profile's career text, score it against "
        "the JD split into positive and negative facets, add interpretable feature "
        "scores, multiply by an availability modifier, then sort and take the top 100.",
        "Models and heuristics: a CPU sentence-transformer (all-MiniLM-L6-v2) for "
        "cosine similarity, a rule-based role classifier, regex evidence "
        "extraction, a gaussian experience-band, and strict honeypot checks.",
        "Combining signals: weighted sum (role 0.34, semantic 0.30, evidence 0.22, "
        "band 0.08, location 0.06) times a behavioral multiplier from 0.55 to 1.08. "
        "Honeypots are forced to zero.",
    ],
    4: [  # Explainability & Data Validation
        "How decisions are explained: every candidate gets a one to two sentence "
        "reason citing real facts, the job title, company, years of experience, the "
        "evidence found, and key signal values.",
        "Preventing hallucination: reasons are generated programmatically from the "
        "exact features that drove the score, so nothing is invented and each one "
        "names an honest concern.",
        "Suspicious profiles: a strict impossibility guard (job duration longer "
        "than time since it started, many expert skills with 0 months used, "
        "end-before-start dates) forces the roughly 80 honeypots to the bottom. "
        "Zero reached our top 100.",
    ],
    7: [  # Results & Performance
        "Ranking quality: the top 100 are 53 recommendation, search and applied-ML "
        "engineers plus 47 ML engineers. Zero keyword-stuffers, zero "
        "research/CV/junior traps, zero honeypots. The top 10 all show production "
        "retrieval and evaluation evidence, 5 to 8 years, in Indian tech hubs.",
        "Constraints: the ranking step runs in about 106 seconds (limit 5 minutes), "
        "under 16 GB RAM, CPU only, with no network and no LLM calls. The heavy "
        "embedding is a one-time offline precompute.",
    ],
    8: [  # Technologies Used
        "Python 3.11 core. sentence-transformers (all-MiniLM-L6-v2) for small, "
        "fast, GPU-free embeddings. NumPy for fast vector math in the timed step. "
        "uv for reproducible environments. Streamlit for the hosted demo. PyMuPDF "
        "for this deck.",
        "Why these: every choice respects the CPU, 5-minute, no-network budget, "
        "keeps the system fully reproducible offline, and stays transparent enough "
        "to defend in the final interview.",
    ],
    9: [  # Submission Assets
        "GitHub repository: github.com/karywnl/fitsignal (full source, README, "
        "single reproduce command).",
        "Live demo: huggingface.co/spaces/karywnl/fitsignal (Streamlit sandbox).",
        "Ranked output: Alonecoder.csv (top 100 candidates, validated).",
        "This deck (PDF) and submission_metadata.yaml (team, compute, AI-tools "
        "declaration).",
    ],
}
# Y at which answers start on each content page (below the printed questions).
START_Y = {1: 165, 2: 190, 3: 190, 4: 190, 7: 165, 8: 165, 9: 135}


def fill_title(page):
    for label, val in [("Team Name :", "Alonecoder"),
                       ("Team Leader Name :", "Karthikeyan M")]:
        rs = page.search_for(label)
        if rs:
            r = rs[0]
            page.insert_text(fitz.Point(r.x1 + 8, r.y1 - 3), val, fontname=FB,
                             fontsize=13, color=DARK)
    rs = page.search_for("Problem Statement :")
    if rs:
        r = rs[0]
        txt = ("From a 100,000-candidate pool, rank the top 100 for a Senior AI "
               "Engineer role by understanding who actually fits, not by matching "
               "keywords.")
        y = r.y1 + 14
        for ln in wrap(txt, 12, 640):
            page.insert_text(fitz.Point(r.x0, y), ln, fontname=F, fontsize=12,
                             color=GREY)
            y += 15


def draw_workflow(page):
    """End-to-end workflow: a left-to-right pipeline wrapped over two rows."""
    y1, y2 = 150, 250
    w, h = 118, 46
    xs = [48, 188, 328, 468, 608]
    row1 = ["Job description + 100k candidate profiles",
            "Career-text docs (skills ignored)",
            "Embeddings, offline one-time (artifacts/)"]
    for i, lab in enumerate(row1):
        r = fitz.Rect(xs[i], y1, xs[i] + w, y1 + h)
        box(page, r, lab)
        if i < len(row1) - 1:
            arrow(page, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
    # connector down to row 2
    arrow(page, (xs[2] + w / 2, y1 + h), (xs[2] + w / 2, y2))
    row2 = ["Ranking step: features + semantic + behavioral",
            "Honeypot guard (impossible profiles to bottom)",
            "Top 100 + faithful reasoning"]
    rx = [xs[2], xs[1], xs[0]]  # go right-to-left under the connector
    for i, lab in enumerate(row2):
        r = fitz.Rect(rx[i], y2, rx[i] + w, y2 + h)
        fill = ACCENTFILL if i == 0 else BOXFILL
        box(page, r, lab, fill=fill)
        if i < len(row2) - 1:
            arrow(page, (rx[i], y2 + h / 2), (rx[i + 1] + w, y2 + h / 2))
    out = fitz.Rect(xs[0], y2 + 78, xs[0] + w, y2 + 78 + 34)
    arrow(page, (xs[0] + w / 2, y2 + h), (xs[0] + w / 2, y2 + 78))
    box(page, out, "submission.csv (validated)", fill=(0.90, 0.97, 0.92),
        edge=(0.45, 0.70, 0.50), font=FB, size=8.6)


def draw_architecture(page):
    """Two-phase architecture: offline embedding feeds a CPU ranking step whose
    five signals merge, pass a honeypot guard, and yield the top 100."""
    # Offline banner
    top = fitz.Rect(48, 100, 672, 132)
    box(page, top, "Offline precompute (one-time, ~50 min):  career text  ->  "
        "all-MiniLM-L6-v2 (CPU)  ->  embeddings .npy in artifacts/",
        fill=ACCENTFILL, edge=PURPLE, font=FB, size=8.8)
    arrow(page, (360, 132), (360, 150), color=BLUE)

    # Ranking-step label
    page.insert_text(fitz.Point(48, 148), "Ranking step  (< 5 min, CPU only, no network):",
                     fontname=FB, fontsize=9, color=BLUE)

    # Five signal boxes on the left
    sigs = ["Semantic match vs JD facets (+ / -)",
            "Role / title classifier (gold-IR, ML, research/CV/junior traps)",
            "Career-text evidence (production retrieval / ranking / eval)",
            "Experience band + location fit",
            "Behavioral availability -> multiplier"]
    sy = 160
    bx, bw, bh, vgap = 48, 250, 34, 9
    centers = []
    for lab in sigs:
        r = fitz.Rect(bx, sy, bx + bw, sy + bh)
        box(page, r, lab, size=8.0)
        centers.append((r.x1, r.y0 + bh / 2))
        sy += bh + vgap

    # Combine box (middle)
    comb = fitz.Rect(360, 210, 470, 262)
    box(page, comb, "Weighted combine", fill=ACCENTFILL, edge=PURPLE, font=FB,
        size=9)
    for cx, cy in centers:
        arrow(page, (cx, cy), (comb.x0, comb.y0 + comb.height / 2), width=1.1)

    # Honeypot guard -> top 100 -> csv
    guard = fitz.Rect(500, 175, 610, 219)
    box(page, guard, "Honeypot guard (force impossible profiles to 0)", size=8.0)
    top100 = fitz.Rect(500, 253, 610, 297)
    box(page, top100, "Top 100 + faithful reasoning", fill=ACCENTFILL, font=FB,
        size=8.4)
    arrow(page, (comb.x1, comb.y0 + 14), (guard.x0, guard.y0 + guard.height / 2))
    arrow(page, (comb.x1, comb.y1 - 14), (top100.x0, top100.y0 + top100.height / 2))
    csv = fitz.Rect(500, 320, 610, 352)
    box(page, csv, "submission.csv", fill=(0.90, 0.97, 0.92),
        edge=(0.45, 0.70, 0.50), font=FB, size=8.6)
    arrow(page, (555, top100.y1), (555, csv.y0))


def draw_results_bar(page):
    """Small stacked bar under the Results answers: 53 gold-IR + 47 ML, 0 traps."""
    x, y, w, h = 360, 250, 300, 16
    page.insert_text(fitz.Point(x, y - 6), "Top 100 composition",
                     fontname=FB, fontsize=8.5, color=GREY)
    page.draw_rect(fitz.Rect(x, y, x + w * 0.53, y + h), color=PURPLE, fill=PURPLE)
    page.draw_rect(fitz.Rect(x + w * 0.53, y, x + w, y + h), color=BLUE, fill=BLUE)
    page.insert_text(fitz.Point(x + 6, y + 11), "53 recsys/search/applied-ML",
                     fontname=FB, fontsize=7.4, color=(1, 1, 1))
    page.insert_text(fitz.Point(x + w * 0.53 + 6, y + 11), "47 ML eng",
                     fontname=FB, fontsize=7.4, color=(1, 1, 1))
    page.insert_text(fitz.Point(x, y + h + 12),
                     "0 keyword-stuffers   |   0 research/CV/junior traps   |   "
                     "0 honeypots", fontname=FB, fontsize=8, color=(0.2, 0.5, 0.3))


def main():
    doc = fitz.open(TEMPLATE)
    fill_title(doc[0])
    for pidx, items in ANSWERS.items():
        page = doc[pidx]
        bullets(page, 72, START_Y[pidx], items, width=600)
    draw_workflow(doc[5])       # p6 End-to-End Workflow
    draw_architecture(doc[6])   # p7 System Architecture
    draw_results_bar(doc[7])    # p8 Results & Performance
    doc.save(OUT, deflate=True)
    print(f"Wrote {OUT} ({doc.page_count} slides)")


if __name__ == "__main__":
    main()
