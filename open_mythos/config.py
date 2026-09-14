"""Configuration management for OpenMythos models."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OpenMythosConfig:
    """Model configuration schema."""

    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    kv_lora_rank: int
    q_lora_rank: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    expert_intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    initializer_range: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]] = None
    architectures: Optional[List[str]] = None

    @classmethod
    def from_json_file(cls, path: str | Path) -> "OpenMythosConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "OpenMythosConfig":
        """Alias for from_json_file()."""
        return cls.from_json_file(path)


ModelConfig = OpenMythosConfig
