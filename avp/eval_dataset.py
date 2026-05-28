#!/usr/bin/env python3
"""
Dataset-wise evaluation using the full plan-execute-replan framework.

Usage:
    python -m avp.eval_dataset --config config.json --limit 10
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import time
import signal
from pathlib import Path
from typing import Any, Dict
import re

# Handle imports when run as script or module
try:
    from .main import Controller, GeminiClient
    from .video_utils import set_metadata_source, cleanup_video_clips, ensure_temp_clips_dir
    from .config import load_config
except ImportError:
    # Running as script, not as module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from avp.main import Controller, GeminiClient
    from avp.video_utils import set_metadata_source, cleanup_video_clips, ensure_temp_clips_dir
    from avp.config import load_config


def format_mcq_query(question: str, options: list[str]) -> str:
    """Format MCQ question with options for better context.
    
    Args:
        question: The question text
        options: List of options (e.g., ["A. Option 1", "B. Option 2", ...])
        
    Returns:
        Formatted query string with question and options
    """
    if not options:
        return question
    
    formatted = f"{question}\n\nOptions:"
    for opt in options:
        formatted += f"\n{opt}"
    return formatted


def extract_answer(final_answer_data: str | Dict[str, Any], options: list[str]) -> str:
    """Extract the selected option from final answer.
    
    Args:
        final_answer_data: Can be a string or a dict from synthesize_final_answer
        options: List of options (for MCQ)
    
    Returns:
        The selected option letter for MCQ, or the answer string for open-ended
    """
    # Handle dict format (from synthesize_final_answer)
    if isinstance(final_answer_data, dict):
        # MCQ format
        if "selected_option" in final_answer_data:
            selected = final_answer_data["selected_option"].strip().upper()
            # If it's already a single letter (A, B, C, D, E, F)
            if len(selected) == 1 and selected.isalpha():
                return selected
            # If it's like "A.", "B.", "C.", extract the letter
            if len(selected) >= 1 and selected[0].isalpha():
                return selected[0]
            return selected
        # Open-ended format
        elif "answer" in final_answer_data:
            return final_answer_data["answer"]
        else:
            # Fallback: return first field as string
            return str(list(final_answer_data.values())[0]) if final_answer_data else ""
    
    # Handle string format (legacy)
    final_answer_upper = str(final_answer_data).upper().strip()
    
    # Try to find option letters
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D
        if letter in final_answer_upper or opt.lower() in str(final_answer_data).lower():
            return letter
    
    # Fallback: return raw answer
    return str(final_answer_data).strip()


def evaluate_dataset(
    ann_path: str,
    out_dir: str,
    config_path: str | None = None,
    limit: int | None = None,
    max_turns: int = 3,
    timeout_per_sample: int | None = None,
    confidence_threshold: float | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Dict[str, Any]:
    """Run full agentic evaluation on video QA dataset.
    
    Args:
        ann_path: Path to annotation JSON file
        out_dir: Output directory for results
        config_path: Optional config file path
        limit: Optional limit on number of samples
        max_turns: Max plan-execute cycles per sample
        timeout_per_sample: Timeout per sample in seconds (None = no timeout)
        
    Returns:
        Summary dict with accuracy and stats
    """
    # Setup
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    if confidence_threshold is not None:
        cfg.confidence_threshold = confidence_threshold
    set_metadata_source(ann_path)
    
    # Load annotations
    with open(ann_path) as f:
        data = json.load(f)
    
    samples = data if isinstance(data, list) else [data]
    if limit and limit > 0:
        samples = samples[:limit]

    # Shard striding for parallel runs: each shard takes every num_shards-th
    # sample (modulo striding balances long/short videos across workers).
    if num_shards > 1:
        if not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_id]
        print(f"[shard {shard_id}/{num_shards}] {len(samples)} samples assigned to this worker")

    print(f"Evaluating {len(samples)} samples with max_turns={max_turns}")
    print(f"Model: {cfg.model}, Project: {cfg.project}, Locations: {cfg.location} (randomly selected per sample)")

    results = []
    correct = 0
    total = 0

    results_file = Path(out_dir) / "results.jsonl"
    
    for idx, sample in enumerate(samples):
        video_id = sample.get("video", sample.get("video_id", f"sample_{idx}"))
        video_path = sample.get("path", sample.get("video_path", ""))
        question = sample.get("question", sample.get("Q", ""))

        ## solution: "<answer>C</answer>",
        solution = sample.get("solution", sample.get("solution", ""))
        ### get the answer from the solution string by regex
        answer = None
        if solution:
            match = re.search(r"<answer>(.*?)</answer>", solution)
            if match:
                answer = match.group(1)
        if not answer:
            print(f"No answer found in solution: {solution}")
            answer = sample.get("answer", sample.get("answer", ""))
        options = sample.get("options", [])
        
        print(f"\n[{idx+1}/{len(samples)}] Video: {video_id}")
        print(f"Question: {question[:100]}...")
        if options:
            print(f"MCQ with {len(options)} options")
        if sample.get("duration"):
            print(f"Duration: {sample.get('duration'):.1f}s")
        
        # Randomly select location for this sample
        sample_location = cfg.get_random_location()
        print(f"Using location: {sample_location}")
        
        try:
            # Import timeout support
            import signal
            
            # Set up timeout signal handler for this sample
            sample_start_time = time.time()
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Sample timed out after {timeout_per_sample}s")
            
            if timeout_per_sample:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_per_sample)
            
            try:
                # Validate video path
                if not video_path:
                    raise ValueError(f"No video path found in sample. video_id={video_id}")
                
                # Ensure temp_clips directory exists for this sample
                try:
                    ensure_temp_clips_dir(video_path, debug=False)
                except Exception as e:
                    print(f"⚠️  Warning: Could not ensure temp_clips directory: {e}")
                    # Continue anyway - create_video_clip will try to create it
                
                # Initialize GeminiClient (API key or Vertex AI)
                api_key_val = (cfg.api_key or "").strip() or None
                client = GeminiClient(
                    model=cfg.model,  # Legacy fallback
                    plan_replan_model=cfg.get_plan_replan_model(),
                    execute_model=cfg.get_execute_model(),
                    project=cfg.project,
                    location=sample_location,
                    api_key=api_key_val,
                    max_frame_low=cfg.max_frame_low,
                    max_frame_medium=cfg.max_frame_medium,
                    max_frame_high=cfg.max_frame_high,
                    confidence_threshold=cfg.confidence_threshold,
                    debug=cfg.debug,
                )
                client.initialize_client()
                
                # Initialize Controller (save in all_sample subfolder)
                sample_dir = str(Path(out_dir) / "all_sample" / f"sample_{idx}")
                Path(sample_dir).mkdir(parents=True, exist_ok=True)
                
                # Save sample metadata for reference
                sample_metadata = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "duration": sample.get("duration"),
                    "time_reference": sample.get("time_reference", ""),
                    "task_type": sample.get("task_type", ""),
                    "full_sample": sample  # Store entire sample for debugging
                }
                with open(Path(sample_dir) / "sample_metadata.json", "w") as f:
                    json.dump(sample_metadata, f, indent=2)
                
                controller = Controller(
                    run_dir=sample_dir,
                    video_path=video_path,
                    client=client,
                    options=options,
                )
                
                # Format query with options for MCQ
                query = format_mcq_query(question, options) if options else question
                
                if options:
                    print(f"Formatted query with {len(options)} MCQ options")
                
                # Run full plan-observe-reflect loop
                result = controller.run(query=query, max_rounds=max_turns)
                final_answer_data = result["final"]
            finally:
                # Cancel the alarm
                if timeout_per_sample:
                    signal.alarm(0)
            
            sample_elapsed = time.time() - sample_start_time
            print(f"Sample completed in {sample_elapsed:.1f}s")
            
            # Extract predicted answer
            pred = extract_answer(final_answer_data, options) if options else extract_answer(final_answer_data, [])

            # Check correctness (for MCQ)
            is_correct = False
            if options and answer:
                # Ground truth might be letter (A) or full text
                if len(answer) == 1 and answer.isalpha():
                    is_correct = pred.upper() == answer.upper()
                else:
                    # Compare with option index
                    try:
                        ans_idx = options.index(answer)
                        ans_letter = chr(65 + ans_idx)
                        is_correct = pred.upper() == ans_letter
                    except (ValueError, IndexError):
                        # Fallback: check if answer text is in the final answer data
                        answer_text = str(final_answer_data).lower()
                        is_correct = answer.lower() in answer_text
            else:
                # Open-ended: simple string match in final answer data
                answer_text = str(final_answer_data).lower()
                is_correct = answer.lower() in answer_text

            if is_correct:
                correct += 1
            total += 1

            result_entry = {
                "video_id": video_id,
                "question": question,
                "ground_truth": answer,
                "predicted": pred,
                "final_answer_data": final_answer_data,
                "correct": is_correct,
                "options": options,
                "location_used": sample_location,  # Track which location was used for this sample
                "num_rounds": len(controller.bb.evidences),  # Number of observation rounds
                "total_evidence_items": sum(len(ev.key_evidence) for ev in controller.bb.evidences),  # Total evidence collected
                "query_confidence": controller.bb.query_confidence,  # Final query confidence
            }
            results.append(result_entry)
            
            # Write incrementally
            with open(results_file, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            
            print(f"Predicted: {pred}, Ground Truth: {answer}, Correct: {is_correct}")
            print(f"Running Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

            # Clean up video clips after sample evaluation (only clips created during this sample)
            cleanup_video_clips(clip_paths=controller.client.created_clips, debug=True)
            # Release any videos uploaded via the Gemini File API to keep
            # project storage bounded across long parallel runs.
            try:
                controller.client.cleanup_uploaded_files()
            except Exception as e:
                print(f"⚠️  Could not cleanup uploaded files: {e}")

        except TimeoutError as e:
            print(f"TIMEOUT on sample {idx}: {e}")
            result_entry = {
                "video_id": video_id,
                "question": question,
                "error": str(e),
                "correct": False,
                "timed_out": True,
                "location_used": sample_location,
            }
            results.append(result_entry)
            total += 1
            with open(results_file, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            
            # Clean up video clips after sample evaluation (even on timeout)
            # Try to get clips from controller if available
            clip_paths = []
            try:
                if 'controller' in locals() and hasattr(controller, 'client'):
                    clip_paths = getattr(controller.client, 'created_clips', [])
            except:
                pass
            cleanup_video_clips(clip_paths=clip_paths, debug=True)
            try:
                if 'controller' in locals() and hasattr(controller, 'client'):
                    controller.client.cleanup_uploaded_files()
            except Exception:
                pass

        except Exception as e:
            print(f"ERROR on sample {idx}: {e}")
            result_entry = {
                "video_id": video_id,
                "question": question,
                "error": str(e),
                "correct": False,
                "timed_out": False,
                "location_used": sample_location,
            }
            results.append(result_entry)
            total += 1
            with open(results_file, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            
            # Clean up video clips after sample evaluation (even on error)
            # Try to get clips from controller if available
            clip_paths = []
            try:
                if 'controller' in locals() and hasattr(controller, 'client'):
                    clip_paths = getattr(controller.client, 'created_clips', [])
            except:
                pass
            cleanup_video_clips(clip_paths=clip_paths, debug=True)
            try:
                if 'controller' in locals() and hasattr(controller, 'client'):
                    controller.client.cleanup_uploaded_files()
            except Exception:
                pass
    
    # Summary
    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "config": {
            "model": cfg.model,
            "project": cfg.project,
            "locations": cfg.location,  # List of locations (randomly selected per sample)
            "max_turns": max_turns,
        },
    }
    
    summary_file = Path(out_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {correct}/{total} = {100*accuracy:.2f}%")
    print(f"Results written to: {out_dir}")
    print(f"{'='*60}")
    
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate video QA dataset with plan-execute-replan")
    ap.add_argument("--ann", required=True, help="Annotation JSON file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--config", default=None, help="Config JSON file")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate first N samples")
    ap.add_argument("--max-turns", type=int, default=3, help="Max plan-execute cycles per sample")
    ap.add_argument("--timeout", type=int, default=None, help="Timeout per sample in seconds")
    ap.add_argument("--confidence-threshold", type=float, default=None,
                    help="Reflector confidence threshold tau_conf (paper Sec 4.3 = 0.7). Overrides config.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of parallel shards (workers).")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="This worker's shard index in [0, num-shards).")
    args = ap.parse_args()

    out = evaluate_dataset(args.ann, args.out, args.config, args.limit, args.max_turns,
                           timeout_per_sample=args.timeout, confidence_threshold=args.confidence_threshold,
                           num_shards=args.num_shards, shard_id=args.shard_id)
    return out


if __name__ == "__main__":
    main()

