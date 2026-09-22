from typing import Tuple, Dict, Any, Optional

import torch
import pydantic
from torch import nn
from torch import Tensor

from models.common import trunc_normal_init_, resolve_bp_steps
from models.transformer import Transformer, Cache, TransformerConfig
from utils.instrumentation import get_active_probe


class HierarchicalReasoningModelConfig(TransformerConfig):
    half_layers: bool = False

    H_cycles: int
    L_cycles: int

    bp_warmup_ratio: float = 0.0
    bp_min_steps: int = 2
    bp_max_steps: int = 5

    # Backprop through every cycle. Sugar for H_bp_steps=-1, L_bp_steps=-1.
    full_backprop: bool = False
    # Explicit per-axis budget, overriding the bp warmup schedule. -1 means "all cycles".
    # NOTE: L_bp_steps counts the last N L-applications across ALL H cycles, so its max is H*L.
    H_bp_steps: Optional[int] = None
    L_bp_steps: Optional[int] = None

    # Change some Transformer config of H-level
    # TODO: Try asymmetric H and L module, such as different size, hidden dims, architecture, attention type, etc.
    H_override: Dict[str, Any] = {}

    @pydantic.model_validator(mode="after")
    def _validate_bp_policy(self):
        if (self.H_bp_steps is None) != (self.L_bp_steps is None):
            raise ValueError("H_bp_steps and L_bp_steps must be set together: the warmup schedule splits a single "
                             "joint budget across both axes, so pinning one while leaving the other on the schedule "
                             "is ill-defined.")
        if self.full_backprop:
            if self.H_bp_steps is not None or self.L_bp_steps is not None:
                raise ValueError("full_backprop=True is ambiguous with an explicit H_bp_steps/L_bp_steps; set one or the other.")
            self.bp_warmup_ratio = 0.0  # Keep the effective policy honest in the saved all_config.yaml.
        return self


class HierarchicalReasoningModelRecurrentBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.core = Transformer(config)

        # Create cache function
        self.create_cache = self.core.create_cache

    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        # Input injection (add)
        # TODO: Try better alternatives, such as GRU / gating in the following papers
        # Alternatively, "fixed" gating that does not depend on hidden state is also worth trying
        # E.g. only depends on position and index of hidden_states dimension
        # https://arxiv.org/pdf/1910.06764
        # https://arxiv.org/pdf/2202.10447
        
        # TODO: Asymmetric fusion is also worth trying. assign different number of tokens to H and L.
        fused = hidden_states + input_injection
        probe = get_active_probe()
        if probe is not None:
            probe.record_fusion(hidden_states, input_injection, fused)
        return self.core(fused, **kwargs)


class HierarchicalReasoningModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = HierarchicalReasoningModelConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        # Reasoning Layers
        # TODO: Asymmetric.
        self.H_level = HierarchicalReasoningModelRecurrentBlock(TransformerConfig(**(config.model_dump() | config.H_override)))
        self.L_level = HierarchicalReasoningModelRecurrentBlock(config)

        # Config
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps
        self.full_backprop = config.full_backprop
        self.cfg_H_bp_steps = config.H_bp_steps
        self.cfg_L_bp_steps = config.L_bp_steps

        self.hidden_size = config.hidden_size
        self.head_hint = self.H_level.core.head_hint  # Hint for LMHead init (inherit from H)
        
        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0), persistent=True)  # NOTE: hardcoded dtype.
        
        # Create cache function
        self.create_cache = lambda **kwargs: dict(H=[self.H_level.create_cache(**kwargs) for _i in range(self.H_cycles)],
                                                  L=[self.L_level.create_cache(**kwargs) for _i in range(self.H_cycles * self.L_cycles)])

    def _resolve_bp_steps(self, bp_steps: int) -> Tuple[int, int]:
        if self.cfg_H_bp_steps is None:  # Both are None; the config validator enforces they move together.
            # Warmup schedule: prioritize H, and at least 1 is allocated to L.
            H_bp_steps = min(self.H_cycles, bp_steps - 1)
            L_bp_steps = bp_steps - H_bp_steps
        else:
            H_bp_steps, L_bp_steps = self.cfg_H_bp_steps, self.cfg_L_bp_steps

        return resolve_bp_steps(self.H_cycles, self.L_cycles, H_bp_steps, L_bp_steps, self.full_backprop)

    def forward(self, carry: None, x: torch.Tensor, cache: Optional[dict[str, list[list[Cache]]]] = None, bp_steps: int = 2, **seq_info) -> Tuple[None, torch.Tensor]:
        z_H, z_L = x, self.zL_init

        H_bp_steps, L_bp_steps = self._resolve_bp_steps(bp_steps)

        probe = get_active_probe()
        if probe is not None:
            probe.record_scalar("bp/H_bp_steps", H_bp_steps, x.device)
            probe.record_scalar("bp/L_bp_steps", L_bp_steps, x.device)

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                if probe is not None:
                    probe.begin_step("L", k, self.H_cycles * self.L_cycles)
                with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)):
                    z_L_next = self.L_level(z_L, z_H, **seq_info, cache=cache["L"][k] if cache is not None else None)
                if probe is not None:
                    probe.record("L", k, z_L, z_L_next, x)
                    probe.end_step()
                z_L = z_L_next
            
            if probe is not None:
                probe.begin_step("H", i, self.H_cycles)
            with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)):
                z_H_next = self.H_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache is not None else None)
            if probe is not None:
                probe.record("H", i, z_H, z_H_next, x)
                probe.end_step()
            z_H = z_H_next

        return None, z_H

    def compute_train_extra_args(self, train_state: Any) -> dict[str, Any]:
        warmup_steps = train_state.total_steps * self.bp_warmup_ratio
        progress = min(1.0, train_state.step / warmup_steps) if warmup_steps > 0 else 1.0

        return dict(bp_steps=self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps)))

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
