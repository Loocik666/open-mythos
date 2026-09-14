"""Tests for Recurrent Depth Transformer architecture."""

from pathlib import Path
import pytest
import torch
from open_mythos.config import OpenMythosConfig
from open_mythos.modeling.recurrent_depth_transformer import (
    RecurrentDepthTransformer,
    RecurrentState,
    RecurrentDepthAttention,
    RecurrentDepthTransformerBlock,
)


CONFIG_PATH = Path("configs/model_micro_1.5b.json")


class TestRecurrentDepthTransformer:
    """Test suite for Recurrent Depth Transformer model."""
    
    def test_config_loads(self):
        """Verify configuration loads correctly."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        assert config.vocab_size > 0
        assert config.hidden_size > 0
        assert config.architectures[0] == "RecurrentDepthTransformer"
    
    def test_model_initializes(self):
        """Verify model initializes without errors."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        assert model is not None
        assert isinstance(model, RecurrentDepthTransformer)
    
    def test_forward_pass_basic(self):
        """Test basic forward pass with single sequence."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 8, config.vocab_size)
        assert not torch.isnan(outputs["logits"]).any()
    
    def test_forward_pass_batched(self):
        """Test forward pass with batched sequences."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (4, 16))
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs["logits"].shape == (4, 16, config.vocab_size)
    
    def test_aux_loss_computation(self):
        """Test auxiliary loss computation for MoE load balancing."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.train()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids)
        
        assert "aux_loss" in outputs
        assert outputs["aux_loss"] > 0
        assert not torch.isnan(outputs["aux_loss"])
    
    def test_backward_pass(self):
        """Test gradient computation through recurrent connections."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.train()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        
        outputs = model(input_ids)
        logits = outputs["logits"]
        aux_loss = outputs["aux_loss"]
        
        # Combined loss with auxiliary loss weighting
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), targets.view(-1)
        )
        total_loss = ce_loss + 0.01 * aux_loss
        total_loss.backward()
        
        # Verify gradients flow through recurrent connections
        grad_count = sum(1 for p in model.parameters() 
                        if p.requires_grad and p.grad is not None)
        assert grad_count > 0
        
        # Check that main components have gradients
        assert model.embed_tokens.weight.grad is not None
        assert model.layers[0].self_attn.q_down_proj.weight.grad is not None
    
    def test_generation_basic(self):
        """Test autoregressive text generation."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=10)
        
        assert generated.shape[1] == input_ids.shape[1] + 10
        assert torch.all(generated[:, :4] == input_ids)
    
    def test_generation_with_sampling(self):
        """Test generation with temperature and top-p sampling."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 4))
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens=8,
                temperature=0.8,
                top_p=0.9
            )
        
        assert generated.shape[1] == 12
        assert generated.shape[0] == 2
    
    def test_recurrent_state_module(self):
        """Test standalone RecurrentState module."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        recurrent_state = RecurrentState(config)
        
        current_hidden = torch.randn(2, 8, config.hidden_size)
        
        # First call (no previous state)
        updated, next_state = recurrent_state(current_hidden)
        assert updated.shape == current_hidden.shape
        
        # Second call (with previous state)
        updated2, next_state2 = recurrent_state(updated, next_state)
        assert updated2.shape == current_hidden.shape
    
    def test_recurrent_attention(self):
        """Test RecurrentDepthAttention with KV recurrence."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        attn = RecurrentDepthAttention(config)
        attn.eval()
        
        x = torch.randn(2, 8, config.hidden_size)
        
        # First layer (no previous KV)
        output1, kv1 = attn(x)
        assert output1.shape == x.shape
        assert kv1.shape == (2, 8, config.kv_lora_rank)
        
        # Simulate next layer with recurrent KV
        output2, kv2 = attn(x, prev_kv_state=kv1)
        assert output2.shape == x.shape
    
    def test_layer_scaling(self):
        """Test learnable layer scaling parameters."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        
        # Check layer scaling exists and varies by layer
        for idx, layer in enumerate(model.layers):
            assert hasattr(layer, 'attn_scale')
            assert hasattr(layer, 'moe_scale')
            
            # Scale should decrease with depth
            expected_scale = 1.0 / (2.0 * (idx + 1))
            assert abs(layer.attn_scale.item() - expected_scale) < 1e-6
            assert abs(layer.moe_scale.item() - expected_scale) < 1e-6
    
    def test_causal_masking(self):
        """Test causal attention masking."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Verify no NaN values (would indicate masking issues)
        assert not torch.isnan(outputs["logits"]).any()
        
        # Verify output shape matches input
        assert outputs["logits"].shape[1] == input_ids.shape[1]
    
    def test_parameter_count(self):
        """Test model parameter count is reasonable."""
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = RecurrentDepthTransformer(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be in reasonable range for the configuration
        assert total_params > 1_000_000  # At least 1M params
        assert total_params < 1_000_000_000  # Less than 1B params


class TestRDTvsLegacy:
    """Compare RDT with legacy MythosModel."""
    
    def test_rdt_has_recurrence_legacy_does_not(self):
        """Verify RDT has recurrent components that legacy model lacks."""
        from open_mythos.model import MythosModel
        
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        rdt_model = RecurrentDepthTransformer(config)
        legacy_model = MythosModel(config)
        
        # RDT should have recurrent state modules
        assert hasattr(rdt_model.layers[0], 'recurrent_state')
        
        # Legacy model should not have these
        assert not hasattr(legacy_model.layers[0], 'recurrent_state')
    
    def test_rdt_output_format_matches_legacy(self):
        """Verify RDT maintains compatible output format."""
        from open_mythos.model import MythosModel
        
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        rdt_model = RecurrentDepthTransformer(config)
        legacy_model = MythosModel(config)
        
        rdt_model.eval()
        legacy_model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        with torch.no_grad():
            rdt_logits = rdt_model(input_ids)["logits"]
            legacy_logits = legacy_model(input_ids)
        
        # Both should produce same shape
        assert rdt_logits.shape == legacy_logits.shape
