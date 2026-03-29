#!/usr/bin/env bash
# eval/harness/run_ragas.sh
#
# Runs the RAGAS scoring phase on an existing pipeline checkpoint,
# rotating through Groq API keys if the token limit is hit.
# Scores one question at a time and resumes from checkpoint on rotation.
#
# Usage (from project root, pass the same output path as run_pipeline.sh):
#   bash eval/harness/run_ragas.sh                                  # default output
#   bash eval/harness/run_ragas.sh benchmarks/phase3_parent.json   # custom output

set -euo pipefail

GROQ_KEYS=(
    "GROQ_KEY_REDACTED_1"
    "GROQ_KEY_REDACTED_2"
    "GROQ_KEY_REDACTED_3"
)

OUTPUT="${1:-benchmarks/ragas_results.json}"
CHECKPOINT="${OUTPUT%.json}.checkpoint.json"
RAGAS_CHECKPOINT="${OUTPUT%.json}.ragas_checkpoint.json"
SLEEP_BETWEEN_CYCLES=60

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No pipeline checkpoint found at $CHECKPOINT. Run the pipeline phase first:"
    echo "  bash eval/harness/run_pipeline.sh $OUTPUT"
    exit 1
fi

ragas_scored() {
    if [[ -f "$RAGAS_CHECKPOINT" ]]; then
        python3 -c "import json; data=json.load(open('$RAGAS_CHECKPOINT')); print(len(data))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

key_index=0
cycle_start_scored=0

while true; do
    current_key="${GROQ_KEYS[$key_index % ${#GROQ_KEYS[@]}]}"
    scored_before=$(ragas_scored)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RAGAS scoring — Key slot $(( key_index % ${#GROQ_KEYS[@]} + 1 ))/${#GROQ_KEYS[@]}  (rotation #${key_index})   Scored: ${scored_before}/20"
    echo "  Checking token budget…"
    uv run python eval/harness/check_limits.py "$current_key"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    export GROQ_API_KEY="$current_key"

    set +e
    uv run python scripts/evaluate.py --ragas-only --output "$OUTPUT"
    exit_code=$?
    set -e

    scored_after=$(ragas_scored)

    if [[ $exit_code -eq 0 ]]; then
        echo ""
        echo "✓ RAGAS scoring complete. Results saved to $OUTPUT"
        exit 0
    fi

    echo "  RAGAS run failed (exit $exit_code). Progress: ${scored_before} → ${scored_after} scored."
    key_index=$(( key_index + 1 ))

    if [[ $(( key_index % ${#GROQ_KEYS[@]} )) -eq 0 ]]; then
        if [[ $scored_after -le $cycle_start_scored ]]; then
            echo "  Full cycle with no progress. Waiting ${SLEEP_BETWEEN_CYCLES}s before retrying…"
            sleep $SLEEP_BETWEEN_CYCLES
        fi
        cycle_start_scored=$scored_after
    fi
done
