"""OpenMythos: Advanced Language Model with MoE and LoRA."""

import torch
import torch.nn as nn
from typing import Optional
from open_mythos.config import OpenMythosConfig


class MythosModel(nn.Module):
    """Advanced language model with LoRA attention and Mixture of Experts."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cpu')

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MythosTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


class MythosTransformerBlock(nn.Module):
    """Transformer block: attention + MoE with pre-normalization."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.attention = MythosAttention(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = MixtureOfExperts(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.moe(self.norm2(hidden_states))
        return hidden_states


class MythosAttention(nn.Module):
    """Multi-head attention with Low-Rank Adaptation (LoRA)."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_proj_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        self.kv_proj_a = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_proj_b = nn.Linear(config.kv_lora_rank, self.num_heads * self.head_dim * 2, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj_b(self.q_proj_a(hidden_states))
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj_b(self.kv_proj_a(hidden_states))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.head_dim * 2).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with shared experts."""

    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList([
            FeedForward(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])

        self.shared_experts = nn.ModuleList([
            FeedForward(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_shared_experts)
        ])

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.clone()
        for expert in self.shared_experts:
            output = output + expert(hidden_states)

        router_logits = self.gate(hidden_states)
        router_weights = torch.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            router_weights, self.num_experts_per_tok, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_out = expert(hidden_states)
                expert_weight = ((top_k_indices == i).float() * top_k_weights).sum(dim=-1, keepdim=True)
                output = output + expert_weight * expert_out

        return output


class FeedForward(nn.Module):
    """SwiGLU-based feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
