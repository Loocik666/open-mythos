import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_mythos.config import OpenMythosConfig

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) layer.
    Compresses Key-Value projections into a low-rank latent space (c_KV)
    to drastically reduce KV-cache memory footprint during inference.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank

        # Low-rank Query (Q) projections
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Low-rank Key-Value (c_KV) compression — core memory savings in MLA
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        # 1. Compress and project Query
        q_compressed = self.q_down_proj(x)
        q = self.q_up_proj(q_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Compress input sequence into latent KV vector (c_kv)
        c_kv = self.kv_down_proj(x)

        # 3. Reconstruct K and V on-the-fly from the compressed latent vector
        k = self.k_up_proj(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Scaled Dot-Product Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # 5. Concatenate attention heads and project to hidden_size
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(output), c_kv
