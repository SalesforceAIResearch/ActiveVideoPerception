"""
Lightweight configuration loader for Active Video Perception.

Supports a single JSON file with fields like:
{
  "project": "your-gcp-project",
  "location": ["us-central1", "us-east1", "global"],  // List of locations - randomly selected per sample
  "model": "gemini-2.0-flash-exp",
  "annotation_path": "/path/to/eval.json",
  "output_dir": "/path/to/out",
  "default_media_resolution": "medium",  // low|medium|high
  "prefer_compressed": true,
  "debug": false
}

Note: location can be a single string (converted to list) or a list of strings.
For each sample, use config.get_random_location() to randomly select a location.

Env overrides (if fields missing):
- VERTEX_PROJECT, VERTEX_LOCATION, GEMINI_MODEL, GEMINI_API_KEY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os
import random
from pathlib import Path


@dataclass
class AVPConfig:
    project: str = "your-gcp-project"
    location: List[str] = field(default_factory=lambda: ["global"])  # List of locations to randomly select from
    model: str = "gemini-2.5-pro"  # Legacy field - used as fallback if plan_replan_model/execute_model not set
    plan_replan_model: str = ""  # Model for planning and replanning (empty = use model field)
    execute_model: str = ""  # Model for video inference/execution (empty = use model field)
    api_key: str = ""  # If set, use Google AI API key (e.g. from Google AI Studio); else use Vertex AI. Prefer GEMINI_API_KEY env.
    annotation_path: str = ""
    output_dir: str = ""
    default_media_resolution: str = "medium"  # low|medium|high
    prefer_compressed: bool = True
    debug: bool = False
    
    # Max frame settings for media resolution
    max_frame_low: int = 512
    max_frame_medium: int = 128
    max_frame_high: int = 128

    # Reflector confidence threshold tau_conf (paper Sec 4.3 uses 0.7):
    # AVP halts and extracts the answer when reflector confidence >= this value.
    confidence_threshold: float = 0.7

    def __post_init__(self):
        """Initialize location as list if it's a string."""
        if isinstance(self.location, str):
            self.location = [self.location]
        elif not isinstance(self.location, list):
            raise ValueError(f"location must be a string or list of strings, got {type(self.location)}")
    
    def get_random_location(self) -> str:
        """Randomly select a location from the location list."""
        if not self.location:
            return "global"
        return random.choice(self.location)

    def get_plan_replan_model(self) -> str:
        """Get the model for planning/replanning operations."""
        if self.plan_replan_model:
            return self.plan_replan_model
        return self.model
    
    def get_execute_model(self) -> str:
        """Get the model for execution/inference operations."""
        if self.execute_model:
            return self.execute_model
        return self.model
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AVPConfig":
        cfg = AVPConfig()
        for k in d:
            if hasattr(cfg, k):
                setattr(cfg, k, d[k])
        # Env overrides
        cfg.project = os.getenv("VERTEX_PROJECT", cfg.project)
        env_location = os.getenv("VERTEX_LOCATION")
        if env_location:
            # If env var is set, convert to list (handle comma-separated values)
            cfg.location = [loc.strip() for loc in env_location.split(",") if loc.strip()]
        # Legacy model env var applies to both if not separately specified
        env_model = os.getenv("GEMINI_MODEL")
        if env_model:
            if not cfg.plan_replan_model:
                cfg.plan_replan_model = env_model
            if not cfg.execute_model:
                cfg.execute_model = env_model
            cfg.model = env_model  # Also set legacy field
        # API key: env overrides config so production can use env-only
        cfg.api_key = os.getenv("GEMINI_API_KEY", cfg.api_key or "")
        # Ensure location is properly initialized as a list
        cfg.__post_init__()
        return cfg


def load_config(path: Optional[str]) -> AVPConfig:
    """Load config from JSON file if provided; else env/defaults.

    Args:
        path: Optional path to a JSON config file
    """
    if path is None or str(path).strip() == "":
        return AVPConfig.from_dict({})

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")
    return AVPConfig.from_dict(data)


