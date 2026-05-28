#!/usr/bin/env bash
# Run AVP evaluation on a QA dataset in PARALLEL.
#
# Launches WORKERS independent processes; sample i is handled by worker
# (i % WORKERS). Process-level parallelism (no GIL, isolated failures,
# per-worker logs), then merges all shards and reports combined accuracy.
#
# Setup:  export GEMINI_API_KEY=your_key   (or configure Vertex AI in the config)
#
# Usage:
#   scripts/run_eval_parallel.sh ANN [OUT] [WORKERS] [CONFIG] [MAX_TURNS] [TAU]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${AVP_PYTHON:-python}"

ANN="${1:-}"
OUT="${2:-$REPO_ROOT/runs/eval_parallel_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${3:-8}"
CONFIG="${4:-$REPO_ROOT/configs/config.example.json}"
MAX_TURNS="${5:-3}"
TAU="${6:-0.7}"

if [[ -z "$ANN" ]]; then
  echo "Usage: scripts/run_eval_parallel.sh ANN [OUT] [WORKERS] [CONFIG] [MAX_TURNS] [TAU]" >&2
  exit 1
fi
[[ -f "$ANN" ]] || { echo "ERROR: annotation not found: $ANN" >&2; exit 1; }
mkdir -p "$OUT"

echo "annotation=$ANN  out=$OUT  workers=$WORKERS  config=$CONFIG  max_turns=$MAX_TURNS  tau_conf=$TAU"

pids=()
for ((i=0; i<WORKERS; i++)); do
  mkdir -p "$OUT/shard_$i"
  "$PY" -m avp.eval_dataset \
      --ann "$ANN" --out "$OUT/shard_$i" --config "$CONFIG" \
      --max-turns "$MAX_TURNS" --confidence-threshold "$TAU" \
      --num-shards "$WORKERS" --shard-id "$i" \
      > "$OUT/shard_$i.log" 2>&1 &
  pids+=("$!")
  echo "  launched worker $i (log: $OUT/shard_$i.log)"
done

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then echo "worker $idx OK"; else echo "worker $idx FAILED (see $OUT/shard_$idx.log)"; fail=1; fi
done

# Merge shard results + combined accuracy
"$PY" - "$OUT" "$WORKERS" <<'PY'
import json, sys, os
out, workers = sys.argv[1], int(sys.argv[2])
merged = os.path.join(out, "results.jsonl"); total = correct = 0
with open(merged, "w") as fo:
    for i in range(workers):
        rf = os.path.join(out, f"shard_{i}", "results.jsonl")
        if not os.path.exists(rf):
            print(f"  [warn] missing {rf}"); continue
        for ln in open(rf):
            ln = ln.strip()
            if not ln: continue
            fo.write(ln + "\n"); r = json.loads(ln)
            total += 1; correct += 1 if r.get("correct") else 0
acc = correct / total if total else 0.0
json.dump({"total": total, "correct": correct, "accuracy": acc, "workers": workers},
          open(os.path.join(out, "summary.json"), "w"), indent=2)
print("=" * 60)
print(f"MERGED ACCURACY: {correct}/{total} = {100*acc:.2f}%")
print(f"Merged results : {merged}")
print("=" * 60)
PY
exit $fail
