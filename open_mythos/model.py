import torch
import torch.nn as nn
from typing import Optional
from open_mythos.config import OpenMythosConfig


class MythosModel(nn.Module):
    """
    OpenMythos language model implementation.
    Architecture:
    - Multi-head attention with LoRA projections
    - Mixture of Experts (MoE) feed-forward layers
    - RoPE (Rotary Position Embeddings)
    - RMS normalization
    """

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cpu')

        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [MythosTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Output projection (logits)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
            attention_mask: Optional attention mask
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        return logits


class MythosTransformerBlock(nn.Module):
    """Single transformer block with attention and MoE feed-forward."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention with LoRA
        self.attention = MythosAttention(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Mixture of Experts
        self.moe = MixtureOfExperts(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm residual: attention
        normed = self.norm1(hidden_states)
        attn_out = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_out
        
        # Pre-norm residual: MoE
        normed = self.norm2(hidden_states)
        moe_out = self.moe(normed)
        hidden_states = hidden_states + moe_out
        
        return hidden_states


class MythosAttention(nn.Module):
    """Multi-head attention with LoRA (Low-Rank Adaptation)."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # LoRA projections for query
        self.q_proj_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_proj_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        # LoRA projections for key/value
        self.kv_proj_a = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_proj_b = nn.Linear(config.kv_lora_rank, self.num_heads * self.head_dim * 2, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # LoRA query projection
        q = self.q_proj_b(self.q_proj_a(hidden_states))
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # LoRA key/value projection
        kv = self.kv_proj_b(self.kv_proj_a(hidden_states))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.head_dim * 2).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        return output


class MixtureOfExperts(nn.Module):
    """Mixture of Experts feed-forward layer."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Create expert networks
        self.experts = nn.ModuleList([
            FeedForward(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            FeedForward(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_shared_experts)
        ])
        
        # Router gate
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Pass through shared experts
        shared_output = hidden_states
        for expert in self.shared_experts:
            shared_output = shared_output + expert(hidden_states)
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch, seq_len, num_experts)
        router_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            router_weights, self.num_experts_per_tok, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Apply expert routing
        moe_output = shared_output.clone()
        for i, expert in enumerate(self.experts):
            # Check if this expert is selected for any token
            expert_mask = (top_k_indices == i).any(dim=-1)  # (batch, seq_len)
            if expert_mask.any():
                expert_output = expert(hidden_states)
                # Get weights for this expert
                expert_weight = (top_k_indices == i).float()  # (batch, seq_len, num_experts_per_tok)
                expert_weight = expert_weight * top_k_weights  # Multiply by softmax weights
                expert_weight = expert_weight.sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
                moe_output = moe_output + expert_weight * expert_output
        
        return moe_output


class FeedForward(nn.Module):
    """Simple feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = torch.nn.functional.silu(self.gate_proj(x))
        x = gate * self.up_proj(x)
        return self.down_proj(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
