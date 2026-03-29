#!/usr/bin/env bash
# eval/harness/run_pipeline.sh
#
# Runs the RAG pipeline phase over the 20-question golden dataset,
# rotating through Groq API keys whenever the token-per-day limit is hit.
# Resumes from the last successful checkpoint automatically on each retry.
#
# Must be run from the project root:
#   bash eval/harness/run_pipeline.sh
#
# Once complete, run the RAGAS scoring phase:
#   bash eval/harness/run_ragas.sh

set -euo pipefail

GROQ_KEYS=(
    "GROQ_KEY_REDACTED_1"
    "GROQ_KEY_REDACTED_2"
    "GROQ_KEY_REDACTED_3"
)

CHECKPOINT="benchmarks/ragas_results.checkpoint.json"
TOTAL_QUESTIONS=20
EXTRA_ARGS=("$@")

# Count successfully completed questions (excludes errors)
questions_done() {
    if [[ -f "$CHECKPOINT" ]]; then
        python3 -c "import json; data=json.load(open('$CHECKPOINT')); print(sum(1 for d in data if not d.get('answer','').startswith('[ERROR:')))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

key_index=0
resume_flag=""
cycle_start_done=0
SLEEP_BETWEEN_CYCLES=60  # seconds to wait after a full cycle with no progress

while true; do
    current_key="${GROQ_KEYS[$key_index % ${#GROQ_KEYS[@]}]}"
    done_before=$(questions_done)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Key slot $(( key_index % ${#GROQ_KEYS[@]} + 1 ))/${#GROQ_KEYS[@]}  (rotation #${key_index})   Questions done: ${done_before}/20"
    echo "  Checking token budget…"
    uv run python eval/harness/check_limits.py "$current_key"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    export GROQ_API_KEY="$current_key"

    set +e
    uv run python scripts/evaluate.py --pipeline-only $resume_flag "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    exit_code=$?
    set -e

    done_after=$(questions_done)

    if [[ $done_after -ge $TOTAL_QUESTIONS ]]; then
        echo ""
        echo "✓ All $TOTAL_QUESTIONS questions answered. Now run RAGAS scoring:"
        echo "  bash eval/harness/run_ragas.sh"
        exit 0
    fi

    if [[ $exit_code -eq 0 && $done_after -lt $TOTAL_QUESTIONS ]]; then
        echo "  Pipeline exited cleanly but only $done_after/$TOTAL_QUESTIONS done. Rotating key…"
    fi

    echo ""
    echo "  Run failed (exit $exit_code). Progress: ${done_before} → ${done_after} questions."

    key_index=$(( key_index + 1 ))
    resume_flag="--resume"

    # After a full cycle, check if any progress was made
    if [[ $(( key_index % ${#GROQ_KEYS[@]} )) -eq 0 ]]; then
        if [[ $done_after -le $cycle_start_done ]]; then
            echo "  Full cycle with no progress. Waiting ${SLEEP_BETWEEN_CYCLES}s before retrying…"
            sleep $SLEEP_BETWEEN_CYCLES
        fi
        cycle_start_done=$done_after
    fi
done
