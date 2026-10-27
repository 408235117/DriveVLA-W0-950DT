import math
import os
import torch
try:
    import torch_npu
    from torch_npu import npu_fusion_attention
except ImportError:
    torch_npu = None

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", "3"))
ATTN_MASK_NPU_CACHE = {}

def get_attn_mask_npu(device, seq_len=2048):
    seq_len = max(int(seq_len), 2048)
    key = (device, seq_len)
    if key not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[key] = torch.triu(torch.ones([seq_len, seq_len], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[key]

def npu_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, head_dim=None, **kwargs):
    """Flash attention wrapper (output-only, no lse/seed).

    Accepts 4D BSND [B,S,H,D] (original behavior, unchanged) or 3D BSH
    [B,S,H*D] (EMU3_ATTN_BSH pipeline). For 3D input, ``head_dim`` must be
    passed explicitly to disambiguate the H*D split (e.g. head_dim=128 ->
    head_num = H*D // 128); 4D input ignores it. The GQA kv head count is
    inferred by the kernel from k/v's layout, exactly as before.
    """
    if q.dim() == 3:
        # BSH: [B, S, H*D] — head_num needs an explicit head_dim to split on.
        if head_dim is None:
            raise ValueError(
                "npu_flash_attn_func: 3D (BSH) input requires head_dim= "
                "(H*D split is ambiguous from shape alone)")
        head_num = q.shape[-1] // head_dim
        layout = "BSH"
        scale_dim = head_dim
    else:
        head_num = q.shape[2]
        layout = "BSND"
        scale_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(scale_dim)
    keep_prob = 1.0 - dropout_p
    if not causal:
        return npu_fusion_attention(q, k, v, head_num, layout, keep_prob=keep_prob, scale=softmax_scale)[0]
    else:
        attn_mask_npu = get_attn_mask_npu(q.device, max(q.shape[1], k.shape[1]))
        return npu_fusion_attention(q, k, v, head_num, layout, keep_prob=keep_prob,
                                    scale=softmax_scale, atten_mask=attn_mask_npu, sparse_mode=SPARSE_MODE)[0]

def npu_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q=None, max_seqlen_k=None,
                                dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    keep_prob = 1.0 - dropout_p
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if not causal:
        head_num = q.shape[1]
        return npu_fusion_attention(q, k, v, head_num, pse=None, atten_mask=None,
                                    scale=softmax_scale, keep_prob=keep_prob, input_layout="TND",
                                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()))[0]
    else:
        q_len = max_seqlen_q if max_seqlen_q is not None else int(cu_seqlens_q[-1])
        k_len = max_seqlen_k if max_seqlen_k is not None else int(cu_seqlens_k[-1])
        attn_mask_npu = get_attn_mask_npu(q.device, max(q_len, k_len))
        head_num = q.shape[1]
        return npu_fusion_attention(q, k, v, head_num, pse=None, padding_mask=None,
                                    atten_mask=attn_mask_npu, scale=softmax_scale, keep_prob=keep_prob,
                                    input_layout="TND",
                                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                                    sparse_mode=SPARSE_MODE)[0]
