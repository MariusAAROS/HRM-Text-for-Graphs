from typing import Tuple, Optional, Sequence, Any, NamedTuple, Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from models.common import trunc_normal_init_, unwrap_tensor
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm
from utils.instrumentation import get_active_probe


Carry = dict[str, Any]
CosSin = Tuple[Tensor, Tensor]
AttnType = Literal["causal", "prefixlm"]


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x:   [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    # Use FP32 RoPE, as in Transformers OLMo and FlashAttention
    # 
    # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmo/modular_olmo.py#L139-L152
    # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/rotary.h#L126-L133
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base, **kwargs):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor):
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]

        return self.cos_cached, self.sin_cached


class LinearInit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Parameters
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class ScaledEmbeddingInit(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 **kwargs):
        super().__init__()
        self.scale = 1.0 / init_std

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


class Cache(NamedTuple):
    """A static cache layer that stores the key and value states as static tensors. Built for `torch.compile` support."""
    keys: Tensor
    values: Tensor

    @classmethod
    def create(cls, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, **kwargs):
        return cls(keys=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs),
                   values=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs))


def _sdpa_with_kvcache(q: Tensor, k: Tensor, v: Tensor, cache: "Cache", cache_lengths: Tensor | int, is_causal: bool) -> Tensor:
    # Ampere-compatible replacement for flash_attn_with_kvcache. Writes the new
    # k,v into the static cache at [cache_lengths : cache_lengths + Sq], then
    # attends over the valid cache range with SDPA. q/k/v: [B, Sq, H, D].
    B, Sq, H, D = q.shape
    max_seq_len = cache.keys.shape[1]
    # flash_attn_with_kvcache accepts a scalar cache_seqlens; the prefill path relies on that.
    if isinstance(cache_lengths, Tensor):
        cache_lengths = cache_lengths.reshape(B).to(torch.long)
    else:
        cache_lengths = torch.full((B,), cache_lengths, dtype=torch.long, device=q.device)

    positions = cache_lengths[:, None] + torch.arange(Sq, device=q.device)[None, :]  # [B, Sq]
    idx = positions[..., None, None].expand(B, Sq, H, D)
    cache.keys.scatter_(1, idx, k)
    cache.values.scatter_(1, idx, v)

    kv_pos = torch.arange(max_seq_len, device=q.device)
    mask = (kv_pos[None, :] < (cache_lengths[:, None] + Sq))[:, None, None, :]       # [B, 1, 1, max_seq_len]
    if is_causal:
        mask = mask & (kv_pos[None, None, None, :] <= positions[:, None, :, None])   # [B, 1, Sq, max_seq_len]

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), cache.keys.transpose(1, 2), cache.values.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)  # [B, Sq, H, D]


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, attn_type, init_std_in=None, init_std_out=None, layer_idx=0, n_layers=1, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        self.gqkv_proj = LinearInit(hidden_size, self.head_dim, batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads, ),
                                   bias=False, init_std=init_std_in, **kwargs)
        self.o_proj = LinearInit(head_dim * num_heads, hidden_size,
                                 bias=False, init_std=init_std_out, **kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        # hidden_states, gqkv: [..., seq_len, hidden_size]
        gqkv = self.gqkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2 * self.num_heads + 2 * self.num_key_value_heads)
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=-2)
        # query, key, value: [..., seq_len, num_heads, head_dim]
        # RoPE
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        is_causal = self.attn_type == "causal"
        if cache is None:
            # flash attn (training)
            seq_info = {name: unwrap_tensor(tensor) for name, tensor in seq_info.items()}
            probe = get_active_probe()
            if probe is not None:
                probe.record_attention(self.layer_idx, self.n_layers, query, key, value, gate, is_causal, seq_info)
            attn_output = flash_attn_varlen_prefixlm(query, key, value, is_causal, **seq_info)
        else:
            # A100 fallback (no FA3): write new k,v into the static cache, then attend
            # over the valid range with SDPA. The graphqa eval uses a cache-free
            # generator, so this path is best-effort for the KV-cache engine only.
            attn_output = _sdpa_with_kvcache(query, key, value, cache, cache_lengths, is_causal)

        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(torch.sigmoid(gate) * attn_output, "... h hd -> ... (h hd)")  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std_in=None, init_std_out=None, **kwargs):
        super().__init__()
        self.gate_up_proj = LinearInit(hidden_size, intermediate_size, batch_out_features=(2, ),
                                       bias=False, init_std=init_std_in, **kwargs)
        self.down_proj    = LinearInit(intermediate_size, hidden_size,
                                       bias=False, init_std=init_std_out, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
