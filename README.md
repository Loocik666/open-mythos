# OpenMythos

**Advanced Language Model Framework with Mixture of Experts (MoE) and Low-Rank Adaptation (LoRA)**

## 🎯 Overview

OpenMythos is a cutting-edge language model architecture combining:
- **Mixture of Experts (MoE)**: Efficient scaling through sparse expert routing
- **LoRA Projections**: Parameter-efficient attention mechanisms
- **RoPE**: Rotary position embeddings for better length extrapolation
- **RMS Normalization**: Stable training with improved convergence

## 📦 Architecture

```
OpenMythos Model
├── Embedding Layer (vocab_size → hidden_size)
├── Transformer Blocks (num_hidden_layers)
│   ├── Multi-Head Attention (LoRA)
│   ├── RMS Normalization
│   └── Mixture of Experts (MoE)
├── Final RMS Normalization
└── Output Projection (hidden_size → vocab_size)
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from open_mythos import MythosModel, OpenMythosConfig
from pathlib import Path

# Load configuration
config = OpenMythosConfig.from_json_file("configs/model_micro_1.5b.json")

# Initialize model
model = MythosModel(config)
model.eval()

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 8))
with torch.no_grad():
    logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # (1, 8, vocab_size)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=open_mythos
```

## 📋 Configuration

Models are configured via JSON files in `configs/` directory:

**Key Parameters:**
- `vocab_size`: Vocabulary size
- `hidden_size`: Hidden dimension
- `num_hidden_layers`: Number of transformer blocks
- `num_attention_heads`: Number of attention heads
- `num_experts`: Total number of MoE experts
- `num_experts_per_tok`: Experts active per token
- `num_shared_experts`: Shared experts across all tokens

## 🏗️ Project Structure

```
open-mythos/
├── open_mythos/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   └── model.py              # Model implementation
├── configs/
│   └── model_micro_1.5b.json # Model configuration
├── tests/
│   └── test_model.py         # Unit tests
├── requirements.txt          # Dependencies
├── pyproject.toml            # Project metadata
└── README.md                 # This file
```

## 📝 License

MIT License - See LICENSE file for details
