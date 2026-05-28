#!/usr/bin/env bash
# Run AVP evaluation on a QA dataset (single process).
#
# The annotation JSON must be a list of samples, each with at least:
#   "path"     : path to the video file
#   "question" : the question text
#   "options"  : list of "A. ...", "B. ..." choices   (for multiple-choice)
#   "solution" : "<answer>C</answer>"  (gold letter)  OR  "answer": "C"
#
# Setup (Google AI Studio API key) :  export GEMINI_API_KEY=your_key
#   or use Vertex AI by setting "project" in the config and leaving the key unset.
#
# Usage:
#   scripts/run_eval.sh ANN [OUT] [CONFIG] [LIMIT] [MAX_TURNS] [TAU]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python interpreter (override with AVP_PYTHON=/path/to/python).
PY="${AVP_PYTHON:-python}"

ANN="${1:-}"
OUT="${2:-$REPO_ROOT/runs/eval_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${3:-$REPO_ROOT/configs/config.example.json}"
LIMIT="${4:-0}"          # 0 / empty => all samples
MAX_TURNS="${5:-3}"      # plan-observe-reflect rounds (paper R_max = 3)
TAU="${6:-0.7}"          # reflector confidence threshold tau_conf (paper Sec 4.3)

if [[ -z "$ANN" ]]; then
  echo "Usage: scripts/run_eval.sh ANN [OUT] [CONFIG] [LIMIT] [MAX_TURNS] [TAU]" >&2
  echo "  ANN: path to the annotation JSON (required)" >&2
  exit 1
fi
[[ -f "$ANN" ]] || { echo "ERROR: annotation not found: $ANN" >&2; exit 1; }
mkdir -p "$OUT"

echo "annotation=$ANN  out=$OUT  config=$CONFIG  limit=$LIMIT  max_turns=$MAX_TURNS  tau_conf=$TAU"

LIMIT_ARG=()
[[ -n "$LIMIT" && "$LIMIT" != "0" ]] && LIMIT_ARG=(--limit "$LIMIT")

exec "$PY" -m avp.eval_dataset \
  --ann "$ANN" \
  --out "$OUT" \
  --config "$CONFIG" \
  --max-turns "$MAX_TURNS" \
  --confidence-threshold "$TAU" \
  "${LIMIT_ARG[@]}"
