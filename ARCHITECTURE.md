# Recurrent Depth Transformer (RDT) Architecture

## Overview

The **Recurrent Depth Transformer** is an advanced neural architecture that introduces cross-layer recurrent connections to improve information flow and reasoning capabilities in deep transformer networks. This implementation builds upon the OpenMythos framework, enhancing it with recurrent mechanisms while maintaining compatibility with existing components like MoE and MLA.

## Core Concepts

### 1. Cross-Layer Recurrent States

Unlike standard transformers where each layer operates independently, RDT maintains recurrent hidden states that flow across all layers:

```
Layer 0: h₀ → RDT Block → h₁' → recurrence → h₁
Layer 1: h₁ → RDT Block → h₂' → recurrence → h₂
...
Layer N: hₙ → RDT Block → hₙ₊₁
```

This enables:
- **Deeper information propagation**: Information from early layers can directly influence deeper layers
- **Improved gradient flow**: Recurrent connections provide additional paths for backpropagation
- **Better reasoning**: State accumulation across layers supports multi-step reasoning

### 2. Gated Recurrence Mechanism

Each recurrent connection uses a gating mechanism inspired by GRU/LSTM architectures:

```python
gate = σ(Linear([h_current, h_previous]))
h_updated = (1 - gate) * f(h_current) + gate * f(h_previous)
h_next = h_current + h_updated
```

Benefits:
- **Stable training**: Gates control information flow, preventing vanishing/exploding gradients
- **Adaptive memory**: Network learns when to update vs. maintain state
- **Deep network support**: Enables training of very deep models

### 3. Recurrent KV Cache

The attention mechanism extends MLA (Multi-Head Latent Attention) with recurrent KV updates:

```python
c_kv = compress(hidden_states)
if prev_kv exists:
    kv_gate = σ(Linear([c_kv, prev_kv]))
    c_kv = (1 - kv_gate) * c_kv + kv_gate * prev_kv
K, V = decompress(c_kv)
```

Advantages:
- **Memory efficiency**: Compressed KV representation reduces cache size
- **Cross-layer context**: KV states accumulate information across layers
- **Inference optimization**: Recurrent caching speeds up generation

### 4. Layer Scaling

Learnable scaling parameters stabilize training in deep networks:

```python
scale = 1 / (2 * (layer_idx + 1))
output = attention(norm(x)) * scale + x
```

This ensures deeper layers contribute appropriately without destabilizing training.

## Architecture Components

### RecurrentState Module
Manages hidden state recurrence across layers with gated updates.

### RecurrentDepthAttention
Combines MLA's low-rank projections with recurrent KV state management and RoPE positioning.

### RecurrentDepthTransformerBlock
Single transformer layer integrating:
- Recurrent state module
- Recurrent depth attention
- Sparse MoE router
- Pre-normalization (RMSNorm)
- Layer scaling

### RecurrentDepthTransformer
Full model with:
- Token embeddings
- Stack of recurrent depth blocks
- Final normalization
- Language model head
- Autoregressive generation support

## Comparison with Standard Transformers

| Feature | Standard Transformer | Recurrent Depth Transformer |
|---------|---------------------|----------------------------|
| Layer Communication | Sequential only | Sequential + Recurrent |
| Hidden State | Per-layer | Cross-layer recurrent |
| KV Cache | Independent per layer | Recurrent across layers |
| Gradient Flow | Layer-by-layer | Additional recurrent paths |
| Memory Usage | O(layers × seq_len) | O(seq_len) with recurrence |
| Reasoning Depth | Limited by layer count | Enhanced by state accumulation |

## Mathematical Formulation

For layer `l` with input `h_l`:

1. **Recurrent State Update**:
   ```
   g = σ(W_g · [h_l, s_{l-1}])
   s_l = h_l + W_out · ((1-g) · W_in · h_l + g · W_in · norm(s_{l-1}))
   ```

2. **Attention with Recurrent KV**:
   ```
   c_kv = W_kv_down · s_l
   if l > 0:
       g_kv = σ(W_gate · [c_kv, c_kv_{l-1}])
       c_kv = (1-g_kv) · c_kv + g_kv · c_kv_{l-1}
   K, V = W_up(c_kv)
   Q = W_q(s_l)
   attn = softmax(QK^T/√d)V
   ```

3. **MoE Processing**:
   ```
   experts = top_k_routing(s_l + attn)
   output = s_l + attn_scale · attn + moe_scale · experts
   ```

## Usage Example

```python
from open_mythos import RecurrentDepthTransformer, OpenMythosConfig

config = OpenMythosConfig.from_json_file("configs/model_micro_1.5b.json")
model = RecurrentDepthTransformer(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 128))
outputs = model(input_ids)
logits = outputs["logits"]
aux_loss = outputs["aux_loss"]

# Training
loss = cross_entropy(logits.view(-1, vocab_size), targets) + 0.01 * aux_loss
loss.backward()

# Generation
generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
```

## Performance Characteristics

### Parameter Efficiency
- Low-rank projections reduce attention parameters
- Sparse MoE activates only subset of experts
- Recurrent sharing reduces per-layer parameters

### Memory Efficiency  
- Compressed KV cache: O(kv_lora_rank) vs O(hidden_size)
- Recurrent states shared across layers
- Efficient generation with cached states

### Training Stability
- Gated recurrence prevents gradient issues
- Layer scaling stabilizes deep networks
- Auxiliary loss balances expert utilization

## Future Enhancements

1. **Chunked Recurrence**: Process sequences in chunks with state handoff
2. **Hierarchical Recurrence**: Multiple recurrence timescales
3. **Attention Recurrence**: Direct attention state carryover
4. **Sparse Recurrence**: Selective recurrent connections for efficiency

## References

- Recurrent Depth Transformer concepts inspired by deep learning research on cross-layer connections
- Multi-Head Latent Attention from DeepSeek-VL work
- Mixture of Experts from Switch Transformer and GShard
- RoPE (Rotary Position Embeddings) from RoFormer

## License

MIT License - See main repository for details