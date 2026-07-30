import torch
from torch import Tensor
import numpy as np

from torch.nn.attention.flex_attention import flex_attention


def compute_aux_seq_tensors_scalars(prefix_lens: np.ndarray, causal_lens: np.ndarray, batch_max_tokens: int):
    # Per-token mask tensors for the FlexAttention prefixLM mask:
    #   doc_ids:     document index of each token (-1 for padding)
    #   prefix_ends: global index where the token's document prefix ends (0 for padding)
    total_lens = prefix_lens + causal_lens
    numseqs = int(total_lens.shape[0])
    total_seqlen = int(total_lens.sum())
    doc_starts = np.concatenate([[0], np.cumsum(total_lens)]).astype(np.int32)  # [numseqs + 1]

    doc_ids = np.full(batch_max_tokens, -1, dtype=np.int32)
    prefix_ends = np.zeros(batch_max_tokens, dtype=np.int32)
    if numseqs > 0 and total_seqlen > 0:
        doc_ids[:total_seqlen] = np.repeat(np.arange(numseqs, dtype=np.int32), total_lens)
        prefix_ends[:total_seqlen] = np.repeat(doc_starts[:numseqs] + prefix_lens.astype(np.int32), total_lens)

    # Tensors
    tensors = {
        "prefix_lens": np.pad(prefix_lens, (0, batch_max_tokens - prefix_lens.shape[0])),
        "causal_lens": np.pad(causal_lens, (0, batch_max_tokens - causal_lens.shape[0])),
        "cu_seqlens": np.pad(np.cumsum(total_lens, dtype=np.int32), (1, batch_max_tokens - total_lens.shape[0] - 1)),
        "doc_ids": doc_ids,
        "prefix_ends": prefix_ends,
    }
    # Scalars
    scalars = {"total_seqlen": total_seqlen,
               "numseqs": numseqs,
               "max_seqlen_prefix": int(prefix_lens.max()),
               "max_seqlen_causal": int(causal_lens.max()),
               "max_seqlen_all": int(total_lens.max())}
    return tensors, scalars


_NEG_INF = float("-inf")


def flash_attn_varlen_prefixlm(q: Tensor, k: Tensor, v: Tensor, is_causal: bool,
                               *, doc_ids: Tensor, prefix_ends: Tensor, **_unused) -> Tensor:
    """PrefixLM attention over a packed batch, via FlexAttention (Ampere-compatible).

    q/k/v are packed as [total_tokens, num_heads, head_dim] (implicit batch of 1);
    num_kv_heads == num_heads (no GQA in this repo). Mask (query q, key kv): attend
    iff same document AND ((kv <= q) OR (kv < prefix_end)); the prefix (bidirectional)
    term is dropped when is_causal. This single-pass mask reproduces the original
    two-pass prefixLM: prefix queries attend bidirectionally over the prefix only,
    response queries attend causally over the full sequence. Padding tokens carry
    doc_id == -1 so real tokens never attend to them; the diagonal is always allowed,
    so every softmax row is well-defined.
    """
    qh = q.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, S, D]
    kh = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    vh = v.permute(1, 0, 2).unsqueeze(0).contiguous()

    doc_ids = doc_ids.to(torch.int32)
    prefix_ends = prefix_ends.to(torch.int32)

    def score_mod(score, b, h, q_idx, kv_idx):
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        causal = kv_idx <= q_idx
        if is_causal:
            allow = same_doc & causal
        else:
            allow = same_doc & (causal | (kv_idx < prefix_ends[kv_idx]))
        return torch.where(allow, score, score.new_full((), _NEG_INF))

    out = flex_attention(qh, kh, vh, score_mod=score_mod)  # [1, H, S, D]
    return out.squeeze(0).permute(1, 0, 2).contiguous()    # [S, H, D]


# NOTE: The FlashAttention-3 (Hopper-only) prefixLM implementation was replaced by
# the FlexAttention `flash_attn_varlen_prefixlm` defined above (Ampere-compatible).
