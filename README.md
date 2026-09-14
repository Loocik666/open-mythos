# OpenMythos

**An Open-Source Implementation Inspired by Anthropic's Mythos Architecture**

> **Important Note**: This is an independent, open-source research project inspired by the architectural concepts of Anthropic's Mythos model. This is **not** affiliated with, endorsed by, or connected to Anthropic PBC. All implementations are based on publicly available information and independent research.

---

## 🎯 Overview

OpenMythos is a research-focused language model framework implementing advanced architectural patterns found in modern large-scale language models. Our implementation features:

- **Recurrent Depth Transformer (RDT)**: Cross-layer recurrent connections for enhanced information flow
- **Mixture of Experts (MoE)**: Sparse expert routing for efficient scaling
- **Multi-Head Latent Attention (MLA)**: Compressed KV-cache for memory efficiency
- **RoPE**: Rotary position embeddings for length extrapolation
- **RMSNorm**: Stable normalization for improved training dynamics

## ⚖️ Relationship to Anthropic's Mythos

### What We Know

Anthropic's Mythos is a proprietary model with limited public technical details. Based on community research and publicly available information from [HuggingFace](https://huggingface.co/maidacundo/open-mythos-hf), we understand it incorporates:

- Advanced transformer architectures
- Mixture of Experts (MoE) design
- Optimized attention mechanisms
- Large-scale training methodologies

### What This Project Is

OpenMythos is our **independent interpretation and implementation** of architectural patterns believed to be present in Mythos-like models. Key distinctions:

| Aspect | Anthropic's Mythos | OpenMythos |
|--------|-------------------|------------|
| **Status** | Proprietary, closed-source | Open-source, community-driven |
| **Training Data** | Anthropic's private datasets | Public datasets (user-defined) |
| **Architecture Details** | Undisclosed | Fully documented (see ARCHITECTURE.md) |
| **Core Innovation** | Anthropic's research | RDT implementation (cross-layer recurrence) |
| **Scale** | Production-scale (reportedly large) | Research-scale (1.5B reference) |
| **Purpose** | Commercial/research use | Educational/research exploration |
| **Weights** | Not publicly released | Train your own from scratch |

### Honest Assessment

**We do not claim to replicate Mythos exactly.** Without access to:
- Original architecture specifications
- Training data composition
- Hyperparameter configurations
- Engineering optimizations

...this project represents our best effort to explore similar architectural ideas in an open, reproducible manner.

**Our implementation**: The Recurrent Depth Transformer implements cross-layer recurrent connections, an architectural pattern that may offer advantages in:
- Deeper information propagation
- Improved gradient flow in very deep networks
- More efficient memory usage during inference

## 📦 Architecture Overview

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

For complete architectural details, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Key Innovations

1. **Cross-Layer Recurrence**: Hidden states propagate recurrently across all transformer layers
2. **Recurrent KV Cache**: Compressed key-value states updated across layers, reducing memory
3. **Gated Recurrence**: Learnable gates control information flow for training stability
4. **Layer Scaling**: Automatic scaling factors for deep network stability

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from open_mythos import RecurrentDepthTransformer, OpenMythosConfig

# Load configuration
config = OpenMythosConfig.from_json_file("configs/model_micro_1.5b.json")

# Initialize model
model = RecurrentDepthTransformer(config)
model.eval()

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 128))
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs["logits"]

print(f"Output shape: {logits.shape}")  # (batch, seq_len, vocab_size)

# Text generation
generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_p=0.9)
print(f"Generated {generated.shape[1]} tokens")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=open_mythos

# Specific test suites
pytest tests/test_recurrent_depth.py -v
pytest tests/test_mla.py -v
pytest tests/test_moe_router.py -v
```

All tests currently passing ✅

## 📋 Configuration

Models are configured via JSON files in `configs/`:

**Reference Configuration** (`model_micro_1.5b.json`):
- **Model Type**: `mythos-rdt-1.5b`
- **Parameters**: ~1.5B total
- **Hidden Size**: 2048
- **Layers**: 24
- **Attention Heads**: 16
- **Experts**: 64 total, 8 active per token
- **Shared Experts**: 2
- **KV Compression**: 512 (low-rank)
- **Vocabulary**: 32,000

**Key Configuration Parameters**:
```json
{
  "model_type": "mythos-rdt-1.5b",
  "vocab_size": 32000,
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "kv_lora_rank": 512,
  "q_lora_rank": 1024,
  "num_experts": 64,
  "num_experts_per_tok": 8,
  "num_shared_experts": 2,
  "rope_theta": 10000,
  "rms_norm_eps": 1e-6
}
```

## 🏗️ Project Structure

```
open-mythos/
├── open_mythos/
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # Configuration management
│   ├── model.py                       # Legacy model (standard transformer)
│   └── modeling/
│       ├── __init__.py
│       ├── recurrent_depth_transformer.py  # Main RDT architecture
│       ├── mla_attention.py           # Multi-Head Latent Attention
│       ├── moe_router.py              # Sparse MoE routing
│       ├── rmsnorm.py                 # RMS Normalization
│       └── diff_attention.py          # Differential attention (experimental)
├── configs/
│   ├── model_micro_1.5b.json          # Reference configuration
│   └── grpo_config.yaml               # RLHF training config
├── tests/
│   ├── test_recurrent_depth.py        # RDT unit tests
│   ├── test_mla.py                    # Attention tests
│   ├── test_moe_router.py             # MoE tests
│   └── test_kernels.py                # Kernel tests
├── scripts/
│   ├── convert_weights.py             # Weight conversion utilities
│   └── run_distributed_train.sh       # Distributed training script
├── ARCHITECTURE.md                    # Detailed architecture docs
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── pyproject.toml                     # Project metadata
```

## 🔬 Technical Comparison

| Feature | Standard Transformer | OpenMythos RDT | Potential Advantage |
|---------|---------------------|----------------|---------------------|
| Layer Communication | Sequential only | Sequential + Recurrent | Better gradient flow |
| KV Cache Memory | O(layers × seq_len) | O(seq_len) with recurrence | Lower memory footprint |
| Information Flow | Layer-local | Cross-layer accumulation | Enhanced reasoning depth |
| Expert Routing | Static top-k | Top-k + shared experts | Better knowledge sharing |
| Training Stability | Standard norms | Layer scaling + gated recurrence | Supports deeper networks |
| Inference Speed | Standard | Optimized recurrent caching | Faster long-context generation |

*Note: Actual performance depends on training quality, data, and scale. These are architectural potentials, not guaranteed improvements.*

## 🎓 Research & Development

This project serves as:
- **Educational resource**: Understanding modern LLM architectures
- **Research platform**: Experimenting with architectural variations
- **Community tool**: Open alternative to proprietary models

### Areas for Exploration

1. **Scaling laws**: How does RDT perform at different scales?
2. **Recurrence depth**: Optimal number of layers for recurrent connections
3. **Expert granularity**: Trade-offs between expert count and specialization
4. **Memory efficiency**: Real-world benchmarks vs. standard transformers
5. **Reasoning capabilities**: Impact of cross-layer recurrence on complex tasks

## 📄 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete technical specification of RDT
- **configs/**: Example configurations for different model sizes
- **tests/**: Comprehensive test suite with usage examples

## 🤝 Contributing

Contributions welcome! Areas needing attention:
- [ ] Pre-training pipelines
- [ ] Fine-tuning scripts (SFT, DPO, PPO)
- [ ] Quantization support
- [ ] Multi-GPU training optimization
- [ ] Benchmark evaluations
- [ ] Integration with popular frameworks (vLLM, TGI, etc.)

## ⚠️ Limitations & Disclaimers

1. **Not production-ready**: This is a research implementation requiring significant engineering for production deployment
2. **No pretrained weights**: Users must train from scratch or adapt existing checkpoints
3. **Unproven at scale**: RDT architecture has not been validated at billion+ parameter scales
4. **No affiliation with Anthropic**: This is an independent project without endorsement
5. **Performance varies**: Actual results depend heavily on training data, compute budget, and hyperparameters

## 📜 License

MIT License - See LICENSE file for details

**Attribution**: If you use this code in your research, please cite:
```
OpenMythos Contributors. "OpenMythos: An Open Implementation of Mythos-inspired Architectures." 
GitHub repository, 2024. https://github.com/kyegomez/OpenMythos
```

## 🔗 References & Inspiration

- Anthropic's Mythos model (public information via HuggingFace community)
- DeepSeek-VL: Multi-Head Latent Attention
- Switch Transformer / GShard: Mixture of Experts
- RoFormer: Rotary Position Embeddings
- Various research papers on cross-layer connections and recurrent transformers

---

**Status**: 🧪 Research/Experimental  
**Maintained By**: Open-source community
