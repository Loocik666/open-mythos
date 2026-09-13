## 📁 Project Structure

```text
open-mythos/
├── ARCHITECTURE.md           # Architecture specification & mathematical background
├── README.md                 # Project manifesto, roadmap, and contributor guide
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata & build configuration
│
├── configs/                  # Hyperparameter & scaling configurations
│   ├── model_80b_config.json # Layer sizes, MoE expert counts, MLA params
│   ├── pretrain_config.yaml  # Distributed pretraining (FSDP / Megatron-LM)
│   └── grpo_config.yaml      # Reinforcement Learning (GRPO) config
│
├── open_mythos/              # Core Python package
│   ├── __init__.py
│   ├── modeling/             # Neural network architecture modules
│   │   ├── __init__.py
│   │   ├── model.py          # Entry point: OpenMythosForCausalLM assembly
│   │   ├── mla_attention.py  # Multi-Head Latent Attention (MLA)
│   │   ├── moe_router.py     # Router, Shared & Routed experts
│   │   ├── diff_attention.py # Differential Attention components
│   │   └── rmsnorm.py        # Fast RMSNorm layer
│   │
│   ├── kernels/              # High-performance GPU kernels (Triton / C++)
│   │   ├── mla_kernels.py    # KV-cache decompression & memory optimization
│   │   └── moe_dispatch.py   # Dynamic token dispatch for MoE experts
│   │
│   ├── trainers/             # Training & alignment pipelines
│   │   ├── pretrain_trainer.py # Pretraining pipeline
│   │   └── grpo_trainer.py   # GRPO alignment trainer
│   │
│   └── agentic/              # Autonomous execution & sandbox integration
│       ├── sandbox.py        # Isolated Docker/REPL environment
│       └── evaluator.py      # Automated task completion evaluator
│
├── tests/                    # Unit and integration tests (pytest)
│   ├── test_mla.py           # Attention mechanics verification
│   ├── test_moe_router.py    # Router load-balancing tests
│   └── test_kernels.py       # Triton kernel precision checks
│
└── scripts/                  # Infrastructure & helper scripts
    ├── convert_weights.py    # Weight conversion & quantization (GGUF / FP8)
    └── run_distributed_train.sh # Multi-node distributed execution launcher
