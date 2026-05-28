# Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-visit-9DC3E6)](https://activevideoperception.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.05774-ECA8A7?logo=arxiv)](https://arxiv.org/abs/2512.05774)
[![Video](https://img.shields.io/badge/Video-2--min_demo-FF0000?logo=youtube)](https://www.youtube.com/watch?v=15SxSE1A0Ow)

</div>


[Ziyang Wang](https://ziyangw2000.github.io/)<sup>1,2*</sup>, [Honglu Zhou](https://sites.google.com/view/hongluzhou/)<sup>1</sup>,[Shijie Wang](https://wang-sj16.github.io/)<sup>1</sup>, [Junnan Li](https://scholar.google.com/citations?user=MuUhwi0AAAAJ&hl=en)<sup>1</sup>, [Caiming Xiong](http://cmxiong.com/)<sup>1</sup>, [Silvio Savarese](https://www.salesforce.com/blog/author/silvio-savarese/)<sup>1</sup>, [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)<sup>2</sup>, [Michael S. Ryoo](https://scholar.google.com/citations?user=vcw0TJIAAAAJ&hl=en)<sup>1</sup>, [Juan Carlos Niebles](https://www.niebles.net/)<sup>1</sup>

<sup>1</sup> Salesforce AI Research  
<sup>2</sup> UNC Chapel Hill  
<sup>*</sup> Work done during internship at Salesforce

---

<div align="center">
  <img src="assets/teaser.png" width="500">
</div>

<br>

## Table of Contents
- [Highlights](#highlights)
- [Setup](#setup)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Highlights


**Active Video Perception (AVP)** is an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels.

**Key ideas:**
- Treat long videos as **interactive environments**
- Iteratively **plan → observe → reflect** to seek evidence
- Allocate computation **adaptively** to informative regions
- Improve **grounding, efficiency, and reasoning faithfulness**

AVP consistently improves over strong MLLM backbones and prior agentic frameworks across multiple long video understanding benchmarks.

<div align="center">
  <img src="assets/table_1.png" width="800">
</div>

<br>

<div align="center">
  <img src="assets/vis.png" width="800">
</div>

<br>

---

## Setup

### 1. Create Conda Environment
Create and activate a fresh conda environment with the required Python version:

```bash
conda create -n avp python=3.10 -y
conda activate avp
```

### 2. Install System Dependencies

```bash
conda install -c conda-forge ffmpeg
ffmpeg -version
pip install -r requirements.txt
```

### 3. Configure backend (Google AI Studio or Vertex AI)

Copy the template config and pick a backend:

```bash
cp configs/config.example.json configs/my_config.json
```

**Google AI Studio API key (simplest):**

```bash
export GEMINI_API_KEY=your_key
```

Leave `project` empty in the config; `api_key` can stay empty since `GEMINI_API_KEY` env wins.

**Vertex AI:** set `"project"` (and optionally `"location"`) in the config; leave `GEMINI_API_KEY` unset.

Pick a model in the config (`model`, `plan_replan_model`, `execute_model`) — e.g. `gemini-2.5-pro` or `gemini-2.5-flash`.

### 4. Prepare your annotation file

The evaluator expects a JSON list of samples; each sample needs at minimum:

```json
{
  "path": "/abs/path/to/video.mp4",
  "question": "What happens when ...?",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "solution": "<answer>C</answer>"
}
```

`solution` (`<answer>X</answer>`) or a plain `answer` field provides the gold label. Omit `options` for open-ended questions.

For the benchmarks in the paper, annotation templates are shipped under `avp/eval_anno/`:

| Benchmark        | Template file                       | # samples |
|------------------|-------------------------------------|-----------|
| MINERVA          | `avp/eval_anno/eval_minerva.json`   | 1,473     |
| LVBench          | `avp/eval_anno/eval_lvbench.json`   | 1,549     |
| MLVU             | `avp/eval_anno/eval_mlvu.json`      | 2,175     |
| Video-MME (long) | `avp/eval_anno/eval_videomme.json`  | 2,700     |
| LongVideoBench   | `avp/eval_anno/eval_lvb.json`       | 1,337     |

Download the videos from each benchmark's original source and fill in the `"path"` field for each sample (the `"path"` field in the shipped templates is empty by design).

---
## Evaluation

The framework runs an iterative **plan → observe → reflect** loop (Algorithm 1 in the paper) using paper defaults `max_turns=3` (R_max) and `confidence_threshold=0.7` (τ_conf).

**Quick smoke test (single process, first 10 samples):**

```bash
scripts/run_eval.sh path/to/annotation.json runs/smoke configs/my_config.json 10
```

**Full evaluation (parallel, recommended):**

```bash
scripts/run_eval_parallel.sh path/to/annotation.json runs/full 16 configs/my_config.json
```

Positional args: `ANN [OUT] [WORKERS] [CONFIG] [MAX_TURNS] [TAU]`. With 16 workers the full MINERVA set (~1500 samples) finishes in ~90 minutes.

**Outputs:** per-worker `runs/full/shard_<i>/` directories (with per-sample plan / evidence / final answer / history), then a merged `runs/full/results.jsonl` and `runs/full/summary.json` reporting combined accuracy.

**Monitor while running:**

```bash
tail -f runs/full/shard_0.log
grep -h "Running Accuracy" runs/full/shard_*.log | tail
```

Override the Python interpreter with `AVP_PYTHON=/path/to/python` before invocation. Re-run only the errored samples with `scripts/rerun_eval_errors.sh RUN_DIR ANN` (defaults to retryable errors; pass `ALL=1` for every error).

---

## Citation

If you find our work useful, please cite:

```bibtex
@misc{wang2025activevideoperceptioniterative,
      title={Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding}, 
      author={Ziyang Wang and Honglu Zhou and Shijie Wang and Junnan Li and Caiming Xiong and Silvio Savarese and Mohit Bansal and Michael S. Ryoo and Juan Carlos Niebles},
      year={2025},
      eprint={2512.05774},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05774}, 
}