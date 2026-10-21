"""Optional Ascend-native fused ops for the Qwen2.5-VL / ROSS training path.

Each helper returns ``None`` when the op is unavailable or its env flag is off,
so every call site keeps its original PyTorch path as the fallback and the same
code still runs on CUDA/CPU.

Flags (set before the process starts):

    ROSS_NPU_RMSNORM=1   -> torch_npu.npu_rms_norm
    ROSS_NPU_SWIGLU=1    -> torch_npu.npu_swiglu
    ROSS_NPU_ROPE=1      -> torch_npu.npu_rotary_mul

See npu-port.md Step 14 for the measured A/B results.
"""

import os

import torch

try:
    import torch_npu
except ImportError:  # CUDA/CPU builds
    torch_npu = None


def _npu_ready(op_name: str) -> bool:
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    return hasattr(torch_npu, op_name)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


_ENABLE_RMSNORM = _flag("ROSS_NPU_RMSNORM")
_ENABLE_SWIGLU = _flag("ROSS_NPU_SWIGLU")
_ENABLE_ROPE = _flag("ROSS_NPU_ROPE")

_HAS_RMSNORM = _npu_ready("npu_rms_norm")
_HAS_SWIGLU = _npu_ready("npu_swiglu")
_HAS_ROTARY = _npu_ready("npu_rotary_mul")


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """RMSNorm via npu_rms_norm, or None to use the reference path."""
    if not (_ENABLE_RMSNORM and _HAS_RMSNORM):
        return None
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor):
    """silu(gate) * up via npu_swiglu on the concatenated gate/up input."""
    if not (_ENABLE_SWIGLU and _HAS_SWIGLU):
        return None
    return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)


def fused_rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x * cos + rotate_half(x) * sin via npu_rotary_mul."""
    if not (_ENABLE_ROPE and _HAS_ROTARY):
        return None
    return torch_npu.npu_rotary_mul(x, cos, sin)
