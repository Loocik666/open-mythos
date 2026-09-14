import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class OpenMythosConfig:
    """
    Configuration schema for OpenMythos models.
    Acts purely as a type structure — all values are dynamically populated from JSON files.
    """
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
    def from_json_file(cls, json_file_path: str) -> "OpenMythosConfig":
        """
        Parses a target JSON file and maps all key-value pairs directly to the class fields.
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_file_path: str) -> "OpenMythosConfig":
        """
        Alias for from_json_file() for backwards compatibility.
        """
        return cls.from_json_file(json_file_path)


# Alias for backwards compatibility
ModelConfig = OpenMythosConfig
