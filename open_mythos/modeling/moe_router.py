import torch
import torch.nn as nn
import torch.nn.functional as F
from open_mythos.config import OpenMythosConfig


class SwiGLUExpert(nn.Module):
    """
    Parametric SwiGLU Feed-Forward Network (FFN).
    All layer dimensions are dynamically passed during initialization.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoERouter(nn.Module):
    """
    Dynamic Sparse Mixture-of-Experts (MoE) routing layer.
    Extracts all architectural parameters dynamically from OpenMythosConfig.
    """
    def __init__(self, config: OpenMythosConfig):
        super().__init__()
        # Dynamic parameter extraction from JSON config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        self.intermediate_size = config.expert_intermediate_size

        # Router gating network parameterized by config dimensions
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Dynamic instantiation of Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts)
        ])

        # Dynamic instantiation of Shared Experts (if configured > 0)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUExpert(self.hidden_size, self.intermediate_size)
                for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # 1. Process via static Shared Experts if present
        shared_output = 0
        if self.shared_experts is not None:
            for shared_exp in self.shared_experts:
                shared_output = shared_output + shared_exp(x_flat)

        # 2. Compute dynamic router gating scores
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        # 3. Select dynamic Top-K experts per token
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 4. Dispatch tokens and aggregate outputs
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            token_indices, expert_pos = torch.where(topk_indices == i)
            if token_indices.numel() > 0:
                exp_input = x_flat[token_indices]
                exp_out = self.experts[i](exp_input)
                weight = topk_weights[token_indices, expert_pos].unsqueeze(-1)
                routed_output.index_add_(0, token_indices, exp_out * weight)

        # 5. Calculate auxiliary load balancing loss
        aux_loss = self._compute_load_balancing_loss(routing_weights, topk_indices)

        final_output = (shared_output + routed_output).view(batch_size, seq_len, hidden_dim)
        return final_output, aux_loss

    def _compute_load_balancing_loss(self, routing_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        num_tokens = routing_weights.shape[0]
        P = routing_weights.mean(dim=0)
        mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        f = mask.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        
        return self.num_experts * torch.sum(P * f)
