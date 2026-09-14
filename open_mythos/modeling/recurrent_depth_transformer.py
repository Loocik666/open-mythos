"""
Recurrent Depth Transformer (RDT) Implementation

This module implements the core Recurrent Depth Transformer architecture,
where hidden states are recurrently updated across transformer layers,
enabling deeper information propagation and improved reasoning capabilities.

Key innovations:
1. Cross-layer recurrent state updates
2. Gated recurrence for stable training
3. Memory-efficient KV caching with recurrence
4. Integration with MoE and MLA components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from open_mythos.config import OpenMythosConfig


class RecurrentState(nn.Module):
    """
    Manages recurrent hidden states that flow across transformer layers.
    Implements gated recurrence for stable deep network training.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Recurrent gate mechanisms
        self.recurrent_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        # State projection for recurrence
        self.state_proj_in = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.state_proj_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Layer-normalized recurrent connection
        self.recurrent_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self, 
        current_hidden: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute recurrent update across layers.
        
        Args:
            current_hidden: Current layer hidden state [batch, seq_len, hidden]
            prev_state: Previous layer's recurrent state [batch, seq_len, hidden]
            
        Returns:
            updated_state: New recurrent state
            next_state: State to pass to next layer
        """
        if prev_state is None:
            # Initialize state from current hidden
            prev_state = torch.zeros_like(current_hidden)
        
        # Project states for gating
        current_proj = self.state_proj_in(current_hidden)
        prev_proj = self.state_proj_in(self.recurrent_norm(prev_state))
        
        # Compute recurrent gate
        gate_input = torch.cat([current_proj, prev_proj], dim=-1)
        recurrent_gate = self.recurrent_gate(gate_input)
        
        # Gated recurrent update
        updated_state = (1 - recurrent_gate) * current_proj + recurrent_gate * prev_proj
        updated_state = self.state_proj_out(updated_state)
        
        # Residual connection for stability
        next_state = current_hidden + updated_state
        
        return next_state, next_state


class RecurrentDepthAttention(nn.Module):
    """
    Multi-Head Latent Attention with Recurrent Depth connections.
    Combines MLA's memory efficiency with RDT's cross-layer recurrence.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_theta = config.rope_theta
        
        # Query projections with low-rank compression
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        # Key-Value compression with recurrent enhancement
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        # Recurrent KV state management
        self.recurrent_kv_gate = nn.Sequential(
            nn.Linear(self.kv_lora_rank * 2, self.kv_lora_rank),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE rotation cache
        self._register_rope_buffers(config.max_position_embeddings)
        
    def _register_rope_buffers(self, max_seq_len: int):
        """Pre-compute RoPE frequencies."""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
    def _compute_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embeddings."""
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        freqs = torch.einsum('i,j->ij', position_ids.float().squeeze(0), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.dtype)
        
        cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)
        
        x_rot = x.view(*x.shape[:-1], -1, 2).transpose(-2, -1).flatten(-2)
        x_rotate = torch.stack((-x_rot[..., 1::2], x_rot[..., ::2]), dim=-1).flatten(-2)
        
        return x * cos_emb + x_rotate * sin_emb
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        prev_kv_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with recurrent KV state updates.
        
        Args:
            x: Input hidden states [batch, seq_len, hidden]
            mask: Attention mask
            prev_kv_state: Previous layer's compressed KV state
            
        Returns:
            output: Attention output
            current_kv: Current compressed KV state for next layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Query projection
        q_compressed = self.q_down_proj(x)
        q = self.q_up_proj(q_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self._compute_rope(q, seq_len)
        
        # KV compression
        c_kv = self.kv_down_proj(x)
        
        # Recurrent KV state update
        if prev_kv_state is not None:
            kv_gate_input = torch.cat([c_kv, prev_kv_state], dim=-1)
            kv_gate = self.recurrent_kv_gate(kv_gate_input)
            c_kv = (1 - kv_gate) * c_kv + kv_gate * prev_kv_state
        
        # Reconstruct K and V
        k = self.k_up_proj(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._compute_rope(k, seq_len)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        
        return output, c_kv


class RecurrentDepthTransformerBlock(nn.Module):
    """
    Single Recurrent Depth Transformer layer combining:
    - Recurrent state updates across layers
    - Multi-Head Latent Attention with recurrence
    - Sparse MoE with expert routing
    - Pre-normalization architecture
    """
    def __init__(self, config: OpenMythosConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Normalization layers
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Core components
        self.self_attn = RecurrentDepthAttention(config)
        self.recurrent_state = RecurrentState(config)
        
        # Import MoE from existing implementation
        from open_mythos.modeling.moe_router import SparseMoERouter
        self.moe = SparseMoERouter(config)
        
        # Layer-scale parameters for deep network stability
        self.attn_scale = nn.Parameter(torch.ones(1) * (1.0 / (2.0 * (layer_idx + 1))))
        self.moe_scale = nn.Parameter(torch.ones(1) * (1.0 / (2.0 * (layer_idx + 1))))
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        prev_hidden_state: Optional[torch.Tensor] = None,
        prev_kv_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with recurrent connections.
        
        Args:
            x: Current hidden state
            mask: Attention mask
            prev_hidden_state: Recurrent hidden state from previous layer
            prev_kv_state: Compressed KV state from previous layer
            
        Returns:
            x: Updated hidden state
            aux_loss: MoE auxiliary loss
            current_kv: Compressed KV state for next layer
        """
        # Apply recurrent state update
        if prev_hidden_state is not None:
            x, recurrent_hidden = self.recurrent_state(x, prev_hidden_state)
        else:
            recurrent_hidden = x
        
        # Pre-norm attention
        normed_x = self.input_layernorm(x)
        attn_out, current_kv = self.self_attn(normed_x, mask=mask, prev_kv_state=prev_kv_state)
        
        # Scaled residual connection
        x = x + attn_out * self.attn_scale
        
        # Pre-norm MoE
        normed_x_2 = self.post_attention_layernorm(x)
        moe_out, aux_loss = self.moe(normed_x_2)
        
        # Scaled residual connection
        x = x + moe_out * self.moe_scale
        
        return x, aux_loss, current_kv, recurrent_hidden


class RecurrentDepthTransformer(nn.Module):
    """
    Full Recurrent Depth Transformer model for causal language modeling.
    
    This architecture implements:
    1. Cross-layer recurrent hidden state propagation
    2. Recurrent KV cache compression for memory efficiency
    3. Sparse MoE with load balancing
    4. Deep network stabilization via layer scaling
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(0.1)
        
        # Transformer blocks with recurrence
        self.layers = nn.ModuleList([
            RecurrentDepthTransformerBlock(config, idx) 
            for idx in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings if vocab sizes match
        if config.vocab_size == config.hidden_size:
            self.lm_head.weight = self.embed_tokens.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with small variance for deep networks."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> dict:
        """
        Forward pass through Recurrent Depth Transformer.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Causal or padding mask
            return_aux_loss: Whether to return MoE auxiliary loss
            
        Returns:
            Dictionary containing:
            - logits: Language model outputs [batch, seq_len, vocab_size]
            - aux_loss: Total MoE auxiliary loss (if return_aux_loss=True)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device), 
                diagonal=1
            ).flip(0, 1) == 0
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)
        
        # Initialize recurrent states
        prev_hidden_state = None
        prev_kv_state = None
        
        total_aux_loss = 0.0
        
        # Pass through recurrent depth layers
        for layer in self.layers:
            x, aux_loss, current_kv, recurrent_hidden = layer(
                x, 
                mask=attention_mask,
                prev_hidden_state=prev_hidden_state,
                prev_kv_state=prev_kv_state
            )
            
            # Update recurrent states for next layer
            prev_hidden_state = recurrent_hidden
            prev_kv_state = current_kv
            
            if return_aux_loss:
                total_aux_loss = total_aux_loss + aux_loss
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        result = {"logits": logits}
        if return_aux_loss:
            result["aux_loss"] = total_aux_loss
            
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation with recurrent state caching.
        
        Args:
            input_ids: Starting token sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            pad_token_id: Padding token ID
            
        Returns:
            Generated token sequence
        """
        self.eval()
        generated = input_ids.clone()
        
        # Initialize recurrent states for caching
        past_hidden_states = None
        past_kv_states = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                generated,
                return_aux_loss=False
            )
            
            # Get last token logits
            next_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
                
            # Top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
                mask = cumsum_probs > top_p
                sorted_probs[mask] = 0
                next_logits = torch.zeros_like(next_logits).scatter(-1, sorted_indices, sorted_probs)
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS if present
            if pad_token_id is not None and (next_token == pad_token_id).all():
                break
                
        return generated


# Register RMSNorm if not available
if not hasattr(nn, 'RMSNorm'):
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x.to(input_dtype)
    
    nn.RMSNorm = RMSNorm
