#!/usr/bin/env python3
"""
Parallel evaluation script for Active Video Perception framework.

This script splits the annotation file into chunks and runs multiple evaluation
processes in parallel, then merges the results.

Usage:
    python -m avp.eval_parallel --ann annotation.json --out output_dir --config config.json --num-workers 4

Key features:
- Splits annotation file into N chunks
- Runs N parallel processes
- Aggregates results from all workers
- Handles failures gracefully
- Provides progress tracking
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def split_annotations(ann_path: str, num_workers: int, output_dir: Path) -> List[str]:
    """Split annotation file into chunks for parallel processing.
    
    Args:
        ann_path: Path to annotation JSON file
        num_workers: Number of parallel workers
        output_dir: Output directory to save chunks
        
    Returns:
        List of paths to chunk files
    """
    # Load annotations
    with open(ann_path) as f:
        data = json.load(f)
    
    samples = data if isinstance(data, list) else [data]
    total_samples = len(samples)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Splitting into {num_workers} workers (~{samples_per_worker} samples each)")
    
    # Create chunk files
    chunk_files = []
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        
        if start_idx >= total_samples:
            break
            
        chunk = samples[start_idx:end_idx]
        chunk_file = output_dir / f"chunk_{i}.json"
        
        with open(chunk_file, 'w') as f:
            json.dump(chunk, f, indent=2)
        
        chunk_files.append(str(chunk_file))
        logger.info(f"Chunk {i}: samples {start_idx}-{end_idx-1} -> {len(chunk)} samples")
    
    return chunk_files


def run_worker(
    chunk_file: str,
    worker_id: int,
    output_base: str,
    config_path: Optional[str],
    max_turns: int,
    limit: Optional[int],
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Run evaluation for a single chunk.
    
    Args:
        chunk_file: Path to annotation chunk
        worker_id: Worker ID
        output_base: Base output directory
        config_path: Config file path
        max_turns: Max plan-execute cycles
        limit: Optional limit on samples
        timeout: Timeout per sample in seconds
        
    Returns:
        Worker result dict
    """
    worker_output = f"{output_base}/worker_{worker_id}"
    log_dir = Path(output_base) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Log file paths
    log_file = log_dir / f"worker_{worker_id}.log"
    
    # Build command - use eval_dataset
    cmd = [
        sys.executable, "-u", "-m", "avp.eval_dataset",  # -u flag for unbuffered output
        "--ann", chunk_file,
        "--out", worker_output,
        "--max-turns", str(max_turns)
    ]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    
    logger.info(f"Worker {worker_id}: Starting evaluation of {chunk_file}")
    logger.info(f"Worker {worker_id}: Log file: {log_file}")
    if timeout:
        logger.info(f"Worker {worker_id}: Timeout per sample: {timeout}s")
    
    start_time = time.time()
    
    try:
        # Open log file for writing with no buffering for real-time updates
        with open(log_file, 'w', buffering=1) as log_f:  # Line buffering
            # Write header
            log_f.write(f"Worker {worker_id} Evaluation Log\n")
            log_f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"{'='*60}\n\n")
            log_f.flush()  # Ensure header is written immediately
            
            # Run the evaluation with both log file and capture
            # Using unbuffered I/O for real-time logging
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=log_f,
                text=True,
                check=True
            )
        
        elapsed = time.time() - start_time
        
        # Append summary to log file
        with open(log_file, 'a', buffering=1) as log_f:  # Line buffering
            log_f.write(f"\n{'='*60}\n")
            log_f.write(f"Completed successfully in {elapsed:.1f}s\n")
            log_f.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.flush()  # Ensure summary is written immediately
        
        logger.info(f"Worker {worker_id}: Completed successfully in {elapsed:.1f}s")
        
        # Read last 500 chars of log for summary
        with open(log_file, 'r') as log_f:
            log_content = log_f.read()
        
        return {
            "worker_id": worker_id,
            "chunk_file": chunk_file,
            "success": True,
            "elapsed_seconds": elapsed,
            "output_dir": worker_output,
            "log_file": str(log_file),
            "log_preview": log_content[-500:]  # Last 500 chars
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        # Append error info to log file
        if log_file.exists():
            with open(log_file, 'a', buffering=1) as log_f:  # Line buffering
                log_f.write(f"\n{'='*60}\n")
                log_f.write(f"FAILED after {elapsed:.1f}s\n")
                log_f.write(f"Return code: {e.returncode}\n")
                log_f.write(f"Error: {str(e)}\n")
                log_f.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_f.flush()  # Ensure error info is written immediately
        
        logger.error(f"Worker {worker_id}: Failed after {elapsed:.1f}s")
        logger.error(f"Worker {worker_id}: Return code: {e.returncode}")
        logger.error(f"Worker {worker_id}: See log: {log_file}")
        
        return {
            "worker_id": worker_id,
            "chunk_file": chunk_file,
            "success": False,
            "elapsed_seconds": elapsed,
            "output_dir": worker_output,
            "log_file": str(log_file),
            "error": str(e),
            "returncode": e.returncode
        }


def merge_results(output_base: str, num_workers: int) -> Dict[str, Any]:
    """Merge results from all workers.
    
    Args:
        output_base: Base output directory
        num_workers: Number of workers
        
    Returns:
        Merged results summary
    """
    output_dir = Path(output_base)
    merged_results = []
    total_correct = 0
    total_samples = 0
    
    # Collect results from each worker
    for worker_id in range(num_workers):
        worker_dir = output_dir / f"worker_{worker_id}"
        results_file = worker_dir / "results.jsonl"
        
        if not results_file.exists():
            logger.warning(f"Worker {worker_id}: No results file found")
            continue
        
        # Read results
        with open(results_file) as f:
            worker_results = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"Worker {worker_id}: {len(worker_results)} samples")
        
        for result in worker_results:
            merged_results.append(result)
            if result.get("correct", False):
                total_correct += 1
            total_samples += 1
    
    # Write merged results
    merged_file = output_dir / "results.jsonl"
    with open(merged_file, 'w') as f:
        for result in merged_results:
            f.write(json.dumps(result) + "\n")
    
    # Calculate summary
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    summary = {
        "total": total_samples,
        "correct": total_correct,
        "accuracy": accuracy,
        "num_workers": num_workers,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Merged results: {total_correct}/{total_samples} = {100*accuracy:.2f}%")
    
    return summary


def run_parallel_evaluation(
    ann_path: str,
    output_dir: str,
    config_path: Optional[str],
    num_workers: int,
    max_turns: int,
    limit: Optional[int],
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Run parallel evaluation with N workers.
    
    Args:
        ann_path: Path to annotation JSON
        output_dir: Output directory
        config_path: Config file path
        num_workers: Number of parallel workers
        max_turns: Max plan-execute cycles
        limit: Optional limit on samples
        timeout: Timeout per sample in seconds
        
    Returns:
        Final summary dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting parallel evaluation with {num_workers} workers")
    logger.info(f"Annotation file: {ann_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Split annotations
    chunk_files = split_annotations(ann_path, num_workers, output_path)
    
    # Run workers in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i, chunk_file in enumerate(chunk_files):
            future = executor.submit(
                run_worker,
                chunk_file=chunk_file,
                worker_id=i,
                output_base=output_dir,
                config_path=config_path,
                max_turns=max_turns,
                limit=limit,
                timeout=timeout
            )
            futures.append(future)
        
        # Wait for completion and log results
        worker_results = []
        for future in as_completed(futures):
            worker_results.append(future.result())
    
    elapsed = time.time() - start_time
    
    # Log worker results
    successful = sum(1 for r in worker_results if r["success"])
    logger.info(f"Workers completed: {successful}/{len(worker_results)}")
    logger.info(f"Total time: {elapsed:.1f}s")
    
    # Save worker results
    with open(output_path / "worker_results.json", 'w') as f:
        json.dump(worker_results, f, indent=2)
    
    # Merge results
    if successful > 0:
        summary = merge_results(output_dir, len(chunk_files))
        summary["total_elapsed_seconds"] = elapsed
        summary["workers_successful"] = successful
        summary["workers_total"] = len(chunk_files)
        
        return summary
    else:
        logger.error("All workers failed!")
        return {
            "success": False,
            "error": "All workers failed",
            "worker_results": worker_results
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel evaluation on video QA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--ann", required=True, help="Annotation JSON file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default=None, help="Config JSON file")
    parser.add_argument("--max-turns", type=int, default=3, help="Max plan-execute cycles per sample")
    parser.add_argument("--limit", type=int, default=None, help="Limit total samples (for testing)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per sample in seconds")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Parallel Active Video Perception Evaluation")
    logger.info("="*60)
    
    summary = run_parallel_evaluation(
        ann_path=args.ann,
        output_dir=args.out,
        config_path=args.config,
        num_workers=args.num_workers,
        max_turns=args.max_turns,
        limit=args.limit,
        timeout=args.timeout
    )
    
    if summary.get("success") is not False:
        logger.info("="*60)
        logger.info(f"FINAL RESULTS: {summary.get('correct', 0)}/{summary.get('total', 0)} = {100*summary.get('accuracy', 0):.2f}%")
        logger.info(f"Time: {summary.get('total_elapsed_seconds', 0):.1f}s")
        logger.info("="*60)
    else:
        logger.error("Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

