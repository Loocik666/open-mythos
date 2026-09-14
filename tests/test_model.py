"""Sanity check tests for OpenMythos model architecture."""

from pathlib import Path
import pytest
import torch
from open_mythos.config import OpenMythosConfig
from open_mythos.model import MythosModel


CONFIG_PATH = Path("configs/model_micro_1.5b.json")


class TestModelLoading:
    def test_config_loads(self):
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        assert config.vocab_size > 0
        assert config.hidden_size > 0

    def test_model_initializes(self):
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = MythosModel(config)
        assert model is not None


class TestForwardPass:
    @pytest.fixture
    def model(self):
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        return MythosModel(config)

    def test_basic_forward(self, model):
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (1, 8, model.config.vocab_size)
        assert not torch.isnan(output).any()

    def test_batch_forward(self, model):
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (4, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (4, 16, model.config.vocab_size)


class TestGradients:
    def test_backward_pass(self):
        config = OpenMythosConfig.from_json_file(str(CONFIG_PATH))
        model = MythosModel(config)
        model.train()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), targets.view(-1)
        )
        loss.backward()
        
        grad_count = sum(1 for p in model.parameters() 
                        if p.requires_grad and p.grad is not None)
        assert grad_count > 0
