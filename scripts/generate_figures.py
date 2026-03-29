"""Generate README figures for Quaestor.

Produces four charts that tell the chunking story:
  Act 1 — failure_analysis.png      (Phase 1 broken chunks by type)
  Act 2 — chunk_distribution.png    (Phase 1 fixed-size spike at ceiling)
  Act 3 — hierarchical_dist.png     (Phase 3 hierarchical, no ceiling spike)
  Act 4 — ragas_comparison.png      (Phase 2 vs Phase 3 outcomes + RAGAS scores)

Output: docs/figures/

Usage:
    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from langchain_core.documents import Document
from quaestor.ingestion.loader import load_edgar_submission
from quaestor.ingestion.chunker import chunk_documents

OUT = Path(__file__).parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
}
plt.rcParams.update(STYLE)

BLUE = "#4C72B0"
RED  = "#DD4444"
GREEN = "#2ecc71"
ORANGE = "#e67e22"
GREY = "#95a5a6"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading JPM 10-K filing…")
filing_dir = Path("data/raw/sec_filings/sec-edgar-filings/JPM/10-K")
submission_dirs = sorted(filing_dir.glob("*/"))
if not submission_dirs:
    print("ERROR: JPM filing not found. Run index_documents.py first.")
    sys.exit(1)

docs = load_edgar_submission(submission_dirs[-1])
print(f"  Loaded {len(docs)} pages")

# Phase 1 — fixed-size chunks (512-char, no overlap)
print("Chunking Phase 1 (fixed-size, 512 chars)…")
fixed_chunks = chunk_documents(docs, strategy="fixed", chunk_size=512, chunk_overlap=0)
fixed_lengths = [len(c.page_content) for c in fixed_chunks]
print(f"  {len(fixed_chunks)} chunks")

# Phase 3 — hierarchical children (256-char)
print("Chunking Phase 3 (hierarchical, 256-char children)…")
hier_chunks = chunk_documents(
    docs,
    strategy="hierarchical",
    parent_chunk_size=1024,
    child_chunk_size=256,
    child_chunk_overlap=0,
)
hier_lengths = [len(c.page_content) for c in hier_chunks]
print(f"  {len(hier_chunks)} child chunks")


# ---------------------------------------------------------------------------
# Act 1 — Failure analysis
# ---------------------------------------------------------------------------

def classify_chunk(text: str) -> str:
    """Conservative classifier — only flag clear structural failures."""
    t = text.strip()
    if len(t) < 100:
        return "F3: Micro-chunk\n(<100 chars)"
    # Mid-sentence break: last non-whitespace char is a lowercase letter,
    # indicating the splitter cut mid-word or mid-clause.
    last_char = t[-1] if t else ""
    if last_char.islower():
        return "F1: Mid-sentence\nbreak"
    # Orphaned number: chunk is mostly digits/symbols with little prose
    digits = sum(1 for ch in t if ch.isdigit())
    alpha  = sum(1 for ch in t if ch.isalpha())
    if alpha > 0 and digits / max(alpha + digits, 1) > 0.55:
        return "F2: Orphaned\nnumber"
    # Table fragment: majority of lines are very short (tabular cells)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if len(lines) >= 4:
        short_lines = sum(1 for l in lines if len(l) < 30)
        if short_lines / len(lines) > 0.6:
            return "F4: Table\nfragment"
    return "Healthy"

print("Classifying Phase 1 chunks…")
labels = [classify_chunk(c.page_content) for c in fixed_chunks]

order = ["F1: Mid-sentence\nbreak", "F2: Orphaned\nnumber",
         "F4: Table\nfragment", "F3: Micro-chunk\n(<100 chars)"]
counts = {k: labels.count(k) for k in order}
broken_total = sum(counts.values())
healthy = len(fixed_chunks) - broken_total

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 1 Fixed-Size Chunker: Failure Analysis — JPM 10-K",
             fontsize=13, fontweight="bold", y=1.01)

bars = ax1.barh(list(counts.keys()), list(counts.values()), color=RED, alpha=0.85, height=0.6)
ax1.bar_label(bars, padding=4, fontsize=10)
ax1.set_xlabel("Number of chunks")
ax1.set_title("Broken Chunks by Failure Type")
ax1.set_xlim(0, max(counts.values()) * 1.18)

# add total broken annotation
ax1.axvline(broken_total / len(order), color=RED, linestyle=":", alpha=0.3)
ax1.text(0.97, 0.03, f"Total broken: {broken_total:,}\n({broken_total/len(fixed_chunks)*100:.0f}% of all chunks)",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=9.5, color=RED,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.8))

sizes  = [healthy, broken_total]
colors = [GREEN, RED]
explode = (0, 0.06)
wedges, texts, autotexts = ax2.pie(
    sizes, labels=["Healthy", "Broken"],
    autopct="%1.0f%%", colors=colors, explode=explode,
    startangle=140, textprops={"fontsize": 11},
    wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Overall Chunk Health — JPM 10-K")

plt.tight_layout()
out1 = OUT / "01_failure_analysis.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out1}")


# ---------------------------------------------------------------------------
# Act 2 — Fixed-size distribution
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Phase 1: Fixed-Size Chunk Length Distribution — JPM 10-K",
             fontsize=13, fontweight="bold", y=1.01)

bins = np.linspace(0, 530, 40)
ax1.hist(fixed_lengths, bins=bins, color=BLUE, alpha=0.85, edgecolor="white")
ax1.axvline(512, color=RED, linestyle="--", linewidth=1.8, label="target = 512")
ax1.set_xlabel("Chunk length (characters)")
ax1.set_ylabel("Number of chunks")
ax1.set_title("Length Distribution")
ax1.legend()
ax1.text(0.97, 0.95,
         f"Spike at ceiling:\n{sum(1 for l in fixed_lengths if l >= 500):,} chunks\nat 500+ chars",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.8))

sorted_lengths = sorted(fixed_lengths)
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
ax2.plot(sorted_lengths, cdf, color=BLUE, linewidth=2)
ax2.axvline(512, color=RED, linestyle="--", linewidth=1.8, label="target = 512")
ax2.axvspan(0, 100, alpha=0.08, color=RED, label="micro-chunks (< 100 chars)")
ax2.set_xlabel("Chunk length (characters)")
ax2.set_ylabel("Cumulative fraction")
ax2.set_title("CDF of Chunk Lengths")
ax2.legend(fontsize=9)

plt.tight_layout()
out2 = OUT / "02_fixed_distribution.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out2}")


# ---------------------------------------------------------------------------
# Act 3 — Hierarchical distribution
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Phase 3: Hierarchical Child Chunk Length Distribution — JPM 10-K",
             fontsize=13, fontweight="bold", y=1.01)

bins3 = np.linspace(0, 270, 40)
ax1.hist(hier_lengths, bins=bins3, color=GREEN, alpha=0.85, edgecolor="white")
ax1.axvline(256, color=ORANGE, linestyle="--", linewidth=1.8, label="target = 256")
ax1.set_xlabel("Chunk length (characters)")
ax1.set_ylabel("Number of chunks")
ax1.set_title("Length Distribution")
ax1.legend()
at_ceiling = sum(1 for l in hier_lengths if l >= 250)
ax1.text(0.97, 0.95,
         f"At ceiling: {at_ceiling:,}\n({at_ceiling/len(hier_lengths)*100:.0f}% of chunks)",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ORANGE, alpha=0.8))

sorted_hier = sorted(hier_lengths)
cdf3 = np.arange(1, len(sorted_hier) + 1) / len(sorted_hier)
ax2.plot(sorted_hier, cdf3, color=GREEN, linewidth=2)
ax2.axvline(256, color=ORANGE, linestyle="--", linewidth=1.8, label="target = 256")
ax2.set_xlabel("Chunk length (characters)")
ax2.set_ylabel("Cumulative fraction")
ax2.set_title("CDF of Chunk Lengths")
ax2.legend(fontsize=9)

# Annotation: splits happen at sentence boundaries, not arbitrary positions
mid_sentence = sum(1 for c in hier_chunks if c.page_content.strip() and c.page_content.strip()[-1].islower())
clean_pct = 100 - (mid_sentence / len(hier_chunks) * 100)
ax2.text(0.05, 0.95,
         f"Clean splits: {clean_pct:.0f}% of chunks\nend at sentence boundary\n(vs ~21% for fixed-size)",
         transform=ax2.transAxes, ha="left", va="top", fontsize=9,
         color=GREEN, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN, alpha=0.8))

plt.tight_layout()
out3 = OUT / "03_hierarchical_distribution.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out3}")


# ---------------------------------------------------------------------------
# Act 4 — RAGAS before/after
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 2 → Phase 3: Impact of Parent Context Injection",
             fontsize=13, fontweight="bold", y=1.01)

# Left: pipeline outcomes
outcomes   = ["Gate-refused\n(correct)", "LLM self-refused", "Answered"]
phase2_pct = [60, 30, 10]
phase3_pct = [60,  0, 40]

x = np.arange(len(outcomes))
w = 0.35
b1 = ax1.bar(x - w/2, phase2_pct, w, label="Phase 2\n(child-only, 256 chars)", color=GREY, alpha=0.9)
b2 = ax1.bar(x + w/2, phase3_pct, w, label="Phase 3\n(parent injection, 1024 chars)", color=BLUE, alpha=0.9)

ax1.bar_label(b1, fmt="%d%%", padding=3, fontsize=9)
ax1.bar_label(b2, fmt="%d%%", padding=3, fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(outcomes, fontsize=10)
ax1.set_ylabel("Percentage of questions (%)")
ax1.set_title("Pipeline Outcomes (20-question dataset)")
ax1.set_ylim(0, 78)
ax1.legend(fontsize=9)

# Annotate the key win
ax1.annotate("Self-refusals\neliminated",
             xy=(1 + w/2, 2), xytext=(1.55, 22),
             arrowprops=dict(arrowstyle="->", color=RED),
             color=RED, fontsize=9.5, fontweight="bold")

# Right: RAGAS scores
metrics    = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
phase2_sc  = [0,     0.138, 0,     0    ]
phase3_sc  = [0.367, 0.263, 0.306, 0.500]

x2 = np.arange(len(metrics))
b3 = ax2.bar(x2 - w/2, phase2_sc, w, label="Phase 2", color=GREY, alpha=0.9)
b4 = ax2.bar(x2 + w/2, phase3_sc, w, label="Phase 3", color=GREEN, alpha=0.9)

ax2.bar_label(b3, fmt="%.3f", padding=3, fontsize=9)
ax2.bar_label(b4, fmt="%.3f", padding=3, fontsize=9)
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_ylabel("RAGAS score")
ax2.set_title("RAGAS Scores")
ax2.set_ylim(0, 0.72)
ax2.legend(fontsize=9)

ax2.text(0.97, 0.97,
         "Phase 2 NaN scores\nshown as 0 (no answers\nto evaluate)",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
         color=GREY, style="italic")

plt.tight_layout()
out4 = OUT / "04_ragas_comparison.png"
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out4}")

print("\nAll figures saved to docs/figures/")
