from typing import Literal, Optional
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from pydantic import BaseModel

from models.layers import SwiGLU, AttnType, Attention, Cache, RotaryEmbedding, find_multiple
from utils.instrumentation import get_active_probe


class InitConfig(BaseModel):
    in_std: float

    attn_out_std: float
    ff_out_std: float


class TransformerConfig(BaseModel):
    # Input config
    max_seq_len: int

    # Transformer config
    n_layers: int

    hidden_size: int
    num_heads: int
    expansion: float

    attn_type: AttnType = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"]
    init_std: Optional[float] = None

    # "peri" (Peri-LN) also normalises each sublayer output before the residual add, which bounds every
    # write to RMS 1 and so caps residual growth (pre-norm lets it reach RMS ~300 in the recurrence).
    norm_type: Literal["pre", "post", "peri"]
    norm_eps: float

    pos_emb_type: Literal["rope", "none"]
    rope_theta: Optional[float] = None

    # Recompute layer activations in the backward pass. Needed to fit deep full-backprop recurrences.
    grad_checkpointing: bool = False

    # [Computed properties]
    @property
    def intermediate_size(self):
        # Automatic compute "intermediate_size" from "expansion"
        # NOTE: The formula is to match the number of GLU parameters to a vanilla Transformer with same expansion
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
    
    @property
    def init_config(self):
        match self.init_type:
            case "fixed_normal":
                in_std = attn_out_std = ff_out_std = self.init_std if self.init_std is not None else 0.02  # defaults to 0.02, as in OLMo 2
            case "lecun_normal":
                in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
                ff_out_std = 1.0 / math.sqrt(self.intermediate_size)
            case "megatron":
                in_std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
                attn_out_std = ff_out_std = in_std / math.sqrt(2.0 * self.n_layers)
            case _:
                raise NotImplementedError()
            
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            attn_type=config.attn_type,

            layer_idx=layer_idx,
            n_layers=config.n_layers,

            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.attn_out_std
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.ff_out_std
        )
        
        self.forward = getattr(self, f"_forward_{config.norm_type}")  # Avoid branching logic in "forward" for torch.compile compatibility
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    # [Forward logic]
    def _forward_pre(self, x: Tensor, **seq_info) -> Tensor:  # Pre Norm
        x = x + self.attn(self.norm(x), **seq_info)
        return x + self.mlp(self.norm(x))
    
    def _forward_post(self, x: Tensor, **seq_info) -> Tensor:  # Post Norm
        x = self.norm(x + self.attn(x, **seq_info))
        return self.norm(x + self.mlp(x))

    def _forward_peri(self, x: Tensor, **seq_info) -> Tensor:  # Peri Norm (input and output norm)
        x = x + self.norm(self.attn(self.norm(x), **seq_info))
        return x + self.norm(self.mlp(self.norm(x)))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_hint = {"in":  {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}  # Hint for LMHead init

        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.max_seq_len, base=config.rope_theta)

        # Layers
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=_layer_idx) for _layer_idx in range(config.n_layers)])
        self.grad_checkpointing = config.grad_checkpointing

        # Final norm for pre / peri norm (post norm already ends normalised)
        self.norm_f = lambda x: x
        if config.norm_type in ("pre", "peri"):
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

        # Create cache function
        self.create_cache = lambda **kwargs: [Cache.create(**kwargs, num_heads=config.num_heads, head_dim=config.hidden_size // config.num_heads) for _i in range(config.n_layers)]

    def forward(self, x: Tensor, cache: Optional[list[Cache]] = None, **seq_info) -> Tensor:
        seq_info["cos_sin"] = self.rotary_emb(seq_info.pop("position_ids", None)) if hasattr(self, "rotary_emb") else None

        # Recompute would double-write the mutable KV cache, so never checkpoint a cached (decode) pass.
        checkpointing = self.grad_checkpointing and cache is None and torch.is_grad_enabled()
        if checkpointing and seq_info["cos_sin"] is not None:
            # FSDP2 casts buffers inside its forward hooks but those do not re-run during the AC
            # recompute, so pin the RoPE dtype here to keep saved and recomputed metadata identical.
            seq_info["cos_sin"] = tuple(t.to(x.dtype) for t in seq_info["cos_sin"])
        if checkpointing and not torch.compiler.is_compiling() and any(layer._compiled_call_impl is None for layer in self.layers):
            # Eager FlexAttention returns WRONG gradients under non-reentrant checkpoint recompute
            # (off by orders of magnitude, see scripts/validate_bp_steps.py); the compiled kernel is
            # correct. So checkpointing is only allowed inside a compiled step or on compiled blocks.
            raise RuntimeError("grad_checkpointing requires compiled TransformerBlocks (compile_scope=block) or a compiled step.")

        # Forward layers
        for layer_id, layer in enumerate(self.layers):
            layer_cache = cache[layer_id] if cache is not None else None
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, **seq_info, cache=layer_cache, use_reentrant=False)
            else:
                x = layer(x, **seq_info, cache=layer_cache)

        probe = get_active_probe()
        if probe is not None:
            probe.record_prenorm(x)
        return self.norm_f(x)
