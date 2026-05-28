# Evaluation

AVP runs an iterative **plan → observe → reflect** loop (paper Algorithm 1). At
each round a planner proposes a targeted video observation, an observer extracts
timestamped evidence, and a **reflector** (an MLLM) verifies the cumulative
evidence against the query, producing a confidence score and a justification. If
confidence ≥ `tau_conf` it extracts the answer; otherwise the justification
guides the next round. Defaults: `max_turns = 3` (R_max), `tau_conf = 0.7`.

## 1. Configure the backend

Copy and edit a config:

```bash
cp configs/config.example.json configs/my_config.json
```

Choose **one** backend:

- **Google AI Studio API key** — leave `project` empty and export the key:
  ```bash
  export GEMINI_API_KEY=your_key
  ```
- **Vertex AI** — set `"project"` (and optionally `"location"`) in the config and
  leave `GEMINI_API_KEY` unset.

Pick the model in the config (`model` / `plan_replan_model` / `execute_model`),
e.g. `gemini-2.5-pro` or `gemini-2.5-flash`.

## 2. Annotation format

A JSON list of samples; each sample needs:

```json
{
  "path": "/path/to/video.mp4",
  "question": "What happens after ...?",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "solution": "<answer>C</answer>"
}
```

`solution` (`<answer>X</answer>`) or a plain `answer` field provides the gold
label. Omit `options` for open-ended questions.

## 3. Run

Single process:

```bash
scripts/run_eval.sh ANN [OUT] [CONFIG] [LIMIT] [MAX_TURNS] [TAU]
# e.g. quick 10-sample smoke test:
scripts/run_eval.sh data/my_dataset.json runs/smoke configs/my_config.json 10
```

Parallel (recommended for full sets):

```bash
scripts/run_eval_parallel.sh ANN [OUT] [WORKERS] [CONFIG] [MAX_TURNS] [TAU]
# e.g. 16 workers:
scripts/run_eval_parallel.sh data/my_dataset.json runs/full 16 configs/my_config.json
```

Each worker handles every `WORKERS`-th sample; results merge into
`OUT/results.jsonl` and `OUT/summary.json` (combined accuracy printed at the end).

Use `AVP_PYTHON=/path/to/python` to select a specific interpreter.

## Re-running errored samples

A run may have a handful of errors (transient 5xx, blocked responses, oversized
videos). Re-run only the errored samples with:

```bash
scripts/rerun_eval_errors.sh RUN_DIR ANN [WORKERS] [CONFIG] [MAX_TURNS] [TAU]
# default: only retryable errors (5xx / null-response)
ALL=1 scripts/rerun_eval_errors.sh RUN_DIR ANN ...   # re-run every errored sample
```

It rebuilds a filtered annotation of just the errored `(video, question)` pairs,
re-runs them via `scripts/run_eval_parallel.sh`, and writes a merged
`RUN_DIR/results_after_rerun.jsonl` with the updated accuracy.

## Notes

- `tau_conf` is exposed via `--confidence-threshold` (overrides the config).
- Very large / long videos may hit Gemini request limits (inline payload size,
  ~1M input tokens, image count). The codebase already (a) routes videos
  >20 MB through the Gemini File API and (b) clamps `fps × duration ≤
  max_frame_*`. For multi-GB raw videos that exceed the File API's 2 GB
  per-file cap, pre-clip or transcode before evaluation.
