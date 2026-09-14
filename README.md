# OpenMythos

**Advanced Language Model Framework with Recurrent Depth Transformer (RDT), Mixture of Experts (MoE), and Multi-Head Latent Attention (MLA)**

## 🎯 Overview

OpenMythos is a cutting-edge language model architecture combining:
- **Recurrent Depth Transformer (RDT)**: Cross-layer recurrent state propagation for improved reasoning
- **Mixture of Experts (MoE)**: Efficient scaling through sparse expert routing
- **Multi-Head Latent Attention (MLA)**: Memory-efficient KV-cache compression
- **RoPE**: Rotary position embeddings for better length extrapolation
- **RMS Normalization**: Stable training with improved convergence

## 📦 Architecture

```
Recurrent Depth Transformer
├── Embedding Layer (vocab_size → hidden_size)
├── Recurrent Depth Blocks (num_hidden_layers)
│   ├── Recurrent State Module (cross-layer gating)
│   ├── Recurrent Depth Attention (MLA + recurrence)
│   │   ├── Low-rank Query projection
│   │   ├── Compressed KV with recurrent update
│   │   └── RoPE positional encoding
│   ├── RMS Normalization
│   └── Mixture of Experts (MoE)
│       ├── Sparse router with top-k selection
│       └── Shared experts for common knowledge
├── Final RMS Normalization
└── Output Projection (hidden_size → vocab_size)
```

### Key Innovations

1. **Cross-Layer Recurrence**: Hidden states flow recurrently across transformer layers, enabling deeper information propagation
2. **Recurrent KV Cache**: Compressed KV states are updated recurrently across layers, reducing memory footprint
3. **Layer Scaling**: Learnable scaling parameters stabilize training in deep networks
4. **Gated Recurrence**: Sigmoid gates control information flow for stable gradient propagation

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from open_mythos import RecurrentDepthTransformer, OpenMythosConfig
from pathlib import Path

# Load configuration
config = OpenMythosConfig.from_json_file("configs/model_micro_1.5b.json")

# Initialize Recurrent Depth Transformer model
model = RecurrentDepthTransformer(config)
model.eval()

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 8))
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs["logits"]
    
print(f"Output shape: {logits.shape}")  # (1, 8, vocab_size)

# Text generation
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_p=0.9)
print(f"Generated sequence length: {generated.shape[1]}")
```

### Using Legacy Model

```python
from open_mythos import MythosModel

# Legacy transformer without recurrence
model = MythosModel(config)
logits = model(input_ids)
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
- `model_type`: Architecture type (e.g., "mythos-rdt-1.5b")
- `vocab_size`: Vocabulary size
- `hidden_size`: Hidden dimension
- `num_hidden_layers`: Number of transformer blocks
- `num_attention_heads`: Number of attention heads
- `head_dim`: Dimension per attention head
- `kv_lora_rank`: Low-rank KV compression dimension
- `q_lora_rank`: Low-rank query projection dimension
- `num_experts`: Total number of MoE experts
- `num_experts_per_tok`: Experts active per token
- `num_shared_experts`: Shared experts across all tokens
- `expert_intermediate_size`: FFN intermediate dimension
- `rope_theta`: RoPE frequency base
- `rms_norm_eps`: RMSNorm epsilon for numerical stability

## 🏗️ Project Structure

```
open-mythos/
├── open_mythos/
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # Configuration management
│   ├── model.py                       # Legacy model implementation
│   └── modeling/
│       ├── __init__.py
│       ├── recurrent_depth_transformer.py  # RDT architecture (NEW)
│       ├── mla_attention.py           # Multi-Head Latent Attention
│       ├── moe_router.py              # Sparse MoE routing
│       ├── rmsnorm.py                 # RMS Normalization
│       └── diff_attention.py          # Differential attention (WIP)
├── configs/
│   └── model_micro_1.5b.json          # Model configuration
├── tests/
│   └── test_*.py                      # Unit tests
├── requirements.txt                   # Dependencies
├── pyproject.toml                     # Project metadata
└── README.md                          # This file
```

## 🔬 Comparison with Original Mythos

| Feature | Original Mythos | OpenMythos RDT |
|---------|----------------|----------------|
| Core Architecture | Standard Transformer | Recurrent Depth Transformer |
| Cross-layer Communication | None | Gated recurrent states |
| KV Cache | Per-layer | Recurrent compressed across layers |
| Attention | MLA | MLA + Recurrent enhancement |
| MoE | Sparse routing | Sparse + shared experts |
| Position Encoding | RoPE | RoPE with recurrent integration |
| Layer Scaling | Fixed | Learnable (1/(2*(layer_idx+1))) |
| Generation | Standard | Optimized with recurrent caching |

## 📝 License

MIT License - See LICENSE file for details
