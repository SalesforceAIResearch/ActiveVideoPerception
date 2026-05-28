#!/usr/bin/env bash
# Re-run errored samples from a previous AVP evaluation run.
#
# Scans RUN_DIR/shard_*/results.jsonl, picks samples that errored, rebuilds a
# filtered annotation containing just those samples, and re-runs them via
# scripts/run_eval_parallel.sh -- then merges the fresh results back over the
# original error rows.
#
# By default only RETRYABLE errors are re-run (transient 5xx, 502, and
# null/blocked-response JSON crashes). Hard 400s (token / byte / image limits)
# are skipped because they will deterministically fail again until the
# offending video is downsampled or split. Set ALL=1 to re-run everything.
#
# Usage:
#   export GEMINI_API_KEY=your_key
#   scripts/rerun_eval_errors.sh RUN_DIR ANN [WORKERS] [CONFIG] [MAX_TURNS] [TAU]
#   ALL=1 scripts/rerun_eval_errors.sh RUN_DIR ANN ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${AVP_PYTHON:-python}"

RUN_DIR="${1:-}"
ANN="${2:-}"
WORKERS="${3:-8}"
CONFIG="${4:-$REPO_ROOT/configs/config.example.json}"
MAX_TURNS="${5:-3}"
TAU="${6:-0.7}"
ALL="${ALL:-0}"

if [[ -z "$RUN_DIR" || -z "$ANN" ]]; then
  echo "Usage: scripts/rerun_eval_errors.sh RUN_DIR ANN [WORKERS] [CONFIG] [MAX_TURNS] [TAU]" >&2
  echo "  RUN_DIR: the previous run's output directory (contains shard_*/results.jsonl)" >&2
  echo "  ANN    : the original annotation JSON used for that run" >&2
  exit 1
fi
[[ -n "${GEMINI_API_KEY:-}" ]] || { echo "ERROR: set GEMINI_API_KEY" >&2; exit 1; }
[[ -d "$RUN_DIR" ]] || { echo "ERROR: RUN_DIR not found: $RUN_DIR" >&2; exit 1; }
[[ -f "$ANN" ]]     || { echo "ERROR: annotation not found: $ANN" >&2; exit 1; }

ERR_ANN="$RUN_DIR/errors_ann.json"
RERUN_OUT="$RUN_DIR/rerun"

# 1) Build a filtered annotation of just the errored samples ------------------
"$PY" - "$RUN_DIR" "$ANN" "$ERR_ANN" "$ALL" <<'PY'
import json, glob, sys, os
run_dir, ann_path, err_ann, allflag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"

def retryable(e: str) -> bool:
    if not e: return False
    if "NoneType" in e: return True                 # null/blocked response crash
    for t in ("500", "502", "503", "UNAVAILABLE", "INTERNAL", "Bad Gateway", "timed out"):
        if t in e: return True
    return False                                    # 400 size/token/image -> not retryable

errored = {}   # (video_id, question) -> error string
n_all = 0
for f in glob.glob(os.path.join(run_dir, "shard_*", "results.jsonl")):
    for ln in open(f):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln)
        e = r.get("error")
        if not e: continue
        n_all += 1
        if allflag or retryable(e):
            errored[(str(r.get("video_id")), str(r.get("question")))] = e

ann = json.load(open(ann_path))
ann = ann if isinstance(ann, list) else [ann]
def key(s): return (str(s.get("video", s.get("video_id"))), str(s.get("question", s.get("Q"))))
picked = [s for s in ann if key(s) in errored]

json.dump(picked, open(err_ann, "w"))
print(f"errored samples total : {n_all}")
print(f"selected for re-run   : {len(picked)}  ({'ALL' if allflag else 'retryable only'})")
print(f"filtered annotation   : {err_ann}")
PY

N=$("$PY" -c "import json;print(len(json.load(open('$ERR_ANN'))))")
if [[ "$N" -eq 0 ]]; then
  echo "Nothing to re-run. Done."
  exit 0
fi

# 2) Re-run them with the existing parallel runner ---------------------------
rm -rf "$RERUN_OUT"
echo ">>> re-running $N samples into $RERUN_OUT  ($WORKERS workers)"
GEMINI_API_KEY="$GEMINI_API_KEY" "$REPO_ROOT/scripts/run_eval_parallel.sh" \
  "$ERR_ANN" "$RERUN_OUT" "$WORKERS" "$CONFIG" "$MAX_TURNS" "$TAU"

# 3) Merge: original good results + fresh re-run results ----------------------
"$PY" - "$RUN_DIR" "$RERUN_OUT" <<'PY'
import json, glob, os, sys
run_dir, rerun_out = sys.argv[1], sys.argv[2]

def load(globpat):
    rows = []
    for f in glob.glob(globpat):
        for ln in open(f):
            ln = ln.strip()
            if ln: rows.append(json.loads(ln))
    return rows

orig  = load(os.path.join(run_dir,  "shard_*", "results.jsonl"))
rerun = load(os.path.join(rerun_out, "shard_*", "results.jsonl"))
def k(r): return (str(r.get("video_id")), str(r.get("question")))
new = {k(r): r for r in rerun}

final, replaced = [], 0
for r in orig:
    if r.get("error") and k(r) in new:
        final.append(new[k(r)]); replaced += 1
    else:
        final.append(r)

outf = os.path.join(run_dir, "results_after_rerun.jsonl")
with open(outf, "w") as fo:
    for r in final: fo.write(json.dumps(r) + "\n")

tot = len(final)
cor = sum(1 for r in final if r.get("correct"))
err = sum(1 for r in final if r.get("error"))
fixed = sum(1 for r in new.values() if not r.get("error"))
print("=" * 60)
print(f"re-ran/replaced : {replaced}   (now error-free: {fixed})")
print(f"FINAL ACCURACY  : {cor}/{tot} = {100*cor/tot:.2f}%   remaining errors: {err}")
print(f"merged file     : {outf}")
print("=" * 60)
PY
