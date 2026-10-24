"""Optional Ascend-native fused ops for Emu3 training.

Each helper returns None when the op is unavailable or its env flag is off,
so every call site keeps its original PyTorch path as the fallback.

Flags (set before the process starts):
    EMU3_NPU_RMSNORM=1   -> torch_npu.npu_rms_norm
    EMU3_NPU_SWIGLU=1    -> torch_npu.npu_swiglu
    EMU3_NPU_ROPE=1      -> torch_npu.npu_rotary_mul
"""

import os
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _npu_ready(op_name):
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    return hasattr(torch_npu, op_name)


def _flag(name, default="0"):
    return os.getenv(name, default) == "1"


_ENABLE_RMSNORM = _flag("EMU3_NPU_RMSNORM")
_ENABLE_SWIGLU = _flag("EMU3_NPU_SWIGLU")
_ENABLE_ROPE = _flag("EMU3_NPU_ROPE")

_HAS_RMSNORM = _npu_ready("npu_rms_norm")
_HAS_SWIGLU = _npu_ready("npu_swiglu")
_HAS_ROTARY = _npu_ready("npu_rotary_mul")


def fused_rms_norm(x, weight, eps):
    if not (_ENABLE_RMSNORM and _HAS_RMSNORM):
        return None
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


def fused_swiglu(gate, up):
    if not (_ENABLE_SWIGLU and _HAS_SWIGLU):
        return None
    return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)


def fused_rotary_mul(x, cos, sin):
    if not (_ENABLE_ROPE and _HAS_ROTARY):
        return None
    return torch_npu.npu_rotary_mul(x, cos, sin)


def _print_status():
    enabled = []
    if _ENABLE_RMSNORM and _HAS_RMSNORM:
        enabled.append("rmsnorm")
    if _ENABLE_SWIGLU and _HAS_SWIGLU:
        enabled.append("swiglu")
    if _ENABLE_ROPE and _HAS_ROTARY:
        enabled.append("rope")
    if enabled:
        print(f"[NPU] Fused ops enabled: {', '.join(enabled)}")

def fused_add_rms_norm(x, residual, weight, eps):
    """Fused residual Add + RMSNorm using npu_add_rms_norm_v2_functional

    Replaces: residual + x, then RMSNorm(residual + x)
    With: single npu_add_rms_norm_v2_functional kernel

    Returns: (normed_output, residual_out) or None if not enabled
    """
    if os.getenv("EMU3_NPU_ADD_RMSNORM") != "1":
        return None
    try:
        import torch_npu
        # v2_functional returns (rstd, output, x_back)
        # output = RMSNorm(residual + x)
        rstd, output, x_back = torch_npu.npu_add_rms_norm_v2_functional(x, residual, weight, eps)
        return output, x_back
    except Exception as e:
        print(f"[NPU] fused_add_rms_norm failed: {e}")
        return None

