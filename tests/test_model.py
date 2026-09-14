import os
import torch
import pytest
from open_mythos.config import OpenMythosConfig
from open_mythos.modeling.model import OpenMythosForCausalLM


@pytest.fixture
def config():
    config_path = "configs/model_micro_1.5b.json"
    assert os.path.exists(config_path), f"Configuration file missing at {config_path}"
    return OpenMythosConfig.from_json_file(config_path)


def test_config_parsing(config):
    """Verifies that dynamic JSON parsing maps directly to OpenMythosConfig fields."""
    assert config.vocab_size == 102400
    assert config.hidden_size == 2048
    assert config.num_experts == 16
    assert config.num_experts_per_tok == 2


def test_full_model_forward_pass(config):
    """
    Executes a dummy forward pass through OpenMythosForCausalLM.
    Verifies output logits tensor shape and MoE load balancing aux loss.
    """
    model = OpenMythosForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    # Check key presence in output structure
    assert "logits" in outputs
    assert "aux_loss" in outputs

    # Validate output dimensions [batch_size, seq_len, vocab_size]
    expected_logits_shape = (batch_size, seq_len, config.vocab_size)
    assert outputs["logits"].shape == expected_logits_shape, (
        f"Expected logits shape {expected_logits_shape}, but got {outputs['logits'].shape}"
    )

    # Validate load balancing auxiliary loss tensor
    assert isinstance(outputs["aux_loss"], torch.Tensor)
    assert outputs["aux_loss"].ndim == 0  # Scalar loss
