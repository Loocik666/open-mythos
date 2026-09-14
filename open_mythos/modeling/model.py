import torch
import torch.nn as nn
from open_mythos.config import OpenMythosConfig
from open_mythos.modeling.rmsnorm import RMSNorm
from open_mythos.modeling.mla_attention import MultiHeadLatentAttention
from open_mythos.modeling.moe_router import SparseMoERouter


class TransformerBlock(nn.Module):
    """
    Single OpenMythos Transformer layer integrating RMSNorm, 
    Multi-Head Latent Attention (MLA), and Sparse MoE Router.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config)
        self.self_attn = MultiHeadLatentAttention(config)
        self.post_attention_layernorm = RMSNorm(config)
        self.moe = SparseMoERouter(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. Pre-Norm Attention Residual Block
        normed_x = self.input_layernorm(x)
        attn_out, c_kv = self.self_attn(normed_x, mask=mask)
        x = x + attn_out

        # 2. Pre-Norm MoE Feed-Forward Residual Block
        normed_x_2 = self.post_attention_layernorm(x)
        moe_out, aux_loss = self.moe(normed_x_2)
        x = x + moe_out

        return x, aux_loss, c_kv


class OpenMythosForCausalLM(nn.Module):
    """
    Top-level causal language model class dynamically configured via OpenMythosConfig.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.config = config

        # Dynamic embedding layer using config.vocab_size and config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Dynamic sequence of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        x = self.embed_tokens(input_ids)

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss, _ = layer(x, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        return {
            "logits": logits,
            "aux_loss": total_aux_loss
        }
