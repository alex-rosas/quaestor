"""Quaestor — Static Demo App.

A fully offline demo that replays the 20-question golden evaluation dataset.
No Ollama, no Groq API key, no ChromaDB required — all responses are
pre-computed from real pipeline runs and stored in eval/results/.

This app exists because running the live pipeline requires:
  - Ollama running locally (for embeddings)
  - A Groq API key (for the LLM)
  - A built ChromaDB index (~18k chunks)

The static demo lets anyone see real system behaviour instantly.

Run:
    uv run streamlit run app_static.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Quaestor — Static Demo",
    page_icon="📑",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load pre-computed results
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("eval/results")
GOLDEN_PATH = Path("eval/golden_dataset.json")

@st.cache_resource
def load_data():
    golden = json.loads(GOLDEN_PATH.read_text())
    phase2 = json.loads((RESULTS_DIR / "phase2_child_baseline_clean.json").read_text())
    phase3 = json.loads((RESULTS_DIR / "phase3_parent_injection.json").read_text())

    # Index by question text for easy lookup
    p2 = {item["user_input"]: item for item in phase2["dataset"]}
    p3 = {item["user_input"]: item for item in phase3["dataset"]}

    questions = golden["questions"]
    return questions, p2, p3, phase2["scores"], phase3["scores"]

questions, phase2_data, phase3_data, phase2_scores, phase3_scores = load_data()

REFUSAL_PHRASE = "don't have enough information"

TYPE_LABELS = {
    "factual":      ("📊", "Factual",      "Single fact retrievable from one section of the filing."),
    "multi_hop":    ("🔗", "Multi-hop",    "Requires combining evidence from multiple sections."),
    "unanswerable": ("🚫", "Unanswerable", "Information not present in the indexed filing — the confidence gate should fire."),
}

CONFIG_LABELS = {
    "phase2": "Phase 2 — Child-only retrieval (256 chars)",
    "phase3": "Phase 3 — Parent context injection (1024 chars)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_refusal(response: str) -> bool:
    return REFUSAL_PHRASE in response.lower()

def outcome_badge(response: str) -> str:
    if is_refusal(response):
        return "🔒 Not answered"
    return "✅ Answered"

def score_or_na(val) -> str:
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return "—"
    return f"{val:.3f}"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    config = st.radio(
        "Pipeline version",
        options=["phase2", "phase3"],
        format_func=lambda x: CONFIG_LABELS[x],
        index=1,
        help=(
            "**Phase 2** — retrieval uses 256-char child chunks passed directly to the LLM. "
            "High self-refusal rate because fragments lack context.\n\n"
            "**Phase 3** — retrieval still uses 256-char children for precision, "
            "but the LLM receives the 1024-char parent window. "
            "LLM self-refusals eliminated."
        ),
    )

    st.divider()

    st.subheader("📊 Aggregate Results")
    scores = phase2_scores if config == "phase2" else phase3_scores
    data_map = phase2_data if config == "phase2" else phase3_data

    answered = sum(1 for q in questions if not is_refusal(data_map.get(q["question"], {}).get("response", REFUSAL_PHRASE)))
    refused  = len(questions) - answered

    col1, col2 = st.columns(2)
    col1.metric("Answered", f"{answered}/20")
    col2.metric("Not answered", f"{refused}/20")

    st.caption("RAGAS scores (averaged over answered questions only):")
    st.markdown(f"""
| Metric | Score |
|--------|-------|
| Faithfulness | {score_or_na(scores.get('faithfulness'))} |
| Answer Relevancy | {score_or_na(scores.get('answer_relevancy'))} |
| Context Precision | {score_or_na(scores.get('context_precision'))} |
| Context Recall | {score_or_na(scores.get('context_recall'))} |
""")

    st.divider()
    st.caption(
        "This is a **static demo** — all responses are pre-computed from real "
        "pipeline runs on the Apple FY2025 10-K.\n\n"
        "No live backend is running. "
        "See the [live app](https://github.com/alex-rosas/quaestor) for the full interactive version."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📑 Quaestor — Static Demo")
st.caption(
    "Pre-computed responses from the 20-question golden evaluation dataset · "
    "Apple FY2025 10-K · Switch pipeline version in the sidebar"
)

st.info(
    "**How to read this demo:** Each question is answered by the pre-selected pipeline "
    "configuration. The unanswerable questions (Q16–Q20) test whether the system "
    "correctly refuses instead of hallucinating. "
    "Switch between Phase 2 and Phase 3 in the sidebar to see the difference "
    "parent context injection makes.",
    icon="ℹ️",
)

st.divider()

# Group questions by type
type_groups = {"factual": [], "multi_hop": [], "unanswerable": []}
for q in questions:
    type_groups[q["type"]].append(q)

for qtype, qs in type_groups.items():
    icon, label, description = TYPE_LABELS[qtype]
    st.subheader(f"{icon} {label} Questions")
    st.caption(description)

    for q in qs:
        qtext = q["question"]
        data = (phase2_data if config == "phase2" else phase3_data).get(qtext, {})
        response = data.get("response", "No data available.")
        refused = is_refusal(response)

        with st.expander(f"{outcome_badge(response)}  {qtext}", expanded=False):

            col_q, col_a = st.columns([1, 1])

            with col_q:
                st.markdown("**Question**")
                st.markdown(qtext)

                if q.get("ground_truth"):
                    st.markdown("**Ground truth**")
                    st.caption(q["ground_truth"])

            with col_a:
                st.markdown(f"**Response** · {CONFIG_LABELS[config]}")

                if refused:
                    if qtype == "unanswerable":
                        st.success(
                            "🔒 **Correctly refused.**\n\n"
                            "This question asks for information not present in the "
                            "indexed filing. The confidence gate fired and no LLM "
                            "call was made — the system chose not to guess."
                        )
                    else:
                        st.warning(
                            "🔒 **Not answered** — retrieval confidence was too low.\n\n"
                            f"> {response}"
                        )
                else:
                    st.markdown(response)

                # RAGAS scores for this question
                faith  = score_or_na(data.get("faithfulness"))
                ar     = score_or_na(data.get("answer_relevancy"))
                cp     = score_or_na(data.get("context_precision"))
                cr     = score_or_na(data.get("context_recall"))

                if not refused:
                    st.caption(
                        f"Faithfulness: **{faith}** · "
                        f"Relevancy: **{ar}** · "
                        f"Precision: **{cp}** · "
                        f"Recall: **{cr}**"
                    )

            # Retrieved contexts
            contexts = data.get("retrieved_contexts", [])
            if contexts:
                with st.expander("📎 Retrieved passages", expanded=False):
                    for i, ctx in enumerate(contexts, 1):
                        st.markdown(f"**Passage {i}**")
                        st.caption(ctx[:400] + ("…" if len(ctx) > 400 else ""))

    st.divider()
