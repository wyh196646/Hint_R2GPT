import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Union



class MultiBranchLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        num_branches: int = 2,
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear), "Expected an nn.Linear to wrap"
        assert num_branches >= 1, "Need at least one LoRA branch"

        # Keep a *deep* copy so that weight/bias grads still flow to the
        # original parameters owned by the parent model.
        self.linear = deepcopy(linear)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.num_branches = num_branches
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

        # LoRA parameters per branch: A (r × in) and B (out × r)
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(r, self.in_features)) for _ in range(num_branches)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.empty(self.out_features, r)) for _ in range(num_branches)]
        )
        self.reset_parameters()

        # Router – index ∈ [0, num_branches‑1].
        self.register_buffer("router_idx", torch.tensor(0, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------
    def set_router_idx(self, idx: int):
        """Set the active branch. Used by helper ``set_router_idx``."""
        if idx < 0 or idx >= self.num_branches:
            raise ValueError(f"router_idx must be between 0 and {self.num_branches-1}")
        self.router_idx.fill_(idx)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def reset_parameters(self):
        # LoRA init per paper (fan‑in scaling for A, zeros for B)
        for A, B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        idx = int(self.router_idx.item())
        # Low‑rank adaptation:  x →  (x · Aᵀ) · Bᵀ
        lora_out = self.dropout(x @ self.lora_A[idx].T) @ self.lora_B[idx].T
        return base_out + lora_out * self.scale




def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    """Helper to recursively replace a sub‑module in its parent."""
    setattr(parent, child_name, new_module)


def apply_multi_branch_lora(
    model: nn.Module,
    target_keywords: Union[str, List[str]] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc"),
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    num_branches: int = 2,
):
    """Recursively traverse *model* and wrap every ``nn.Linear`` whose
    name **contains** one of *target_keywords* with
    :class:`MultiBranchLoRALinear`.

    Parameters
    ----------
    model : nn.Module
        The root model to be patched (e.g. vision encoder or LLM).
    target_keywords : str | list[str]
        Sub‑strings to match against module names.
    r, alpha, dropout, num_branches : see ``MultiBranchLoRALinear``.
    """
    if isinstance(target_keywords, str):
        target_keywords = [target_keywords]

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(kw in name for kw in target_keywords):
            wrapped = MultiBranchLoRALinear(module, r=r, alpha=alpha, dropout=dropout, num_branches=num_branches)
            _replace_module(model, name, wrapped)
        else:
            apply_multi_branch_lora(module, target_keywords, r, alpha, dropout, num_branches)


def set_router_idx(model: nn.Module, idx: int):
    """Set *router_idx* on **all** :class:`MultiBranchLoRALinear` inside
    *model*. This mirrors the helper used in LION."""
    for m in model.modules():
        if isinstance(m, MultiBranchLoRALinear):
            m.set_router_idx(idx)


# def activate_branch(self, idx: int):
#     """激活指定 LoRA 分支，并冻结其它分支参数。"""
#     set_router_idx(self, idx)
#     for m in self.modules():
#         if isinstance(m, MultiBranchLoRALinear):
#             for j in range(m.num_branches):
#                 for p in (m.lora_A[j], m.lora_B[j]):
#                     p.requires_grad = (j == idx)