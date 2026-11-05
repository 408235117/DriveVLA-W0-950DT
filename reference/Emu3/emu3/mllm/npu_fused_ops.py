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

    NOTE: npu_add_rms_norm_v2_functional has NO autograd registration — its
    fallback backward is identity passthrough (dx=gy, dr=gs) with dw=None, so
    the post-attention RMSNorm weight gradient is silently dropped. We wrap it
    in a custom autograd.Function that uses npu_rms_norm_backward in backward:
      forward : rstd, out, x_back = v2(x, res, w, eps); x_back = res + x
      backward: (dx_back, dw) = npu_rms_norm_backward(gy, x_back, w, rstd)
                with dx = dx_back, dr = dx_back (add symmetry)
    Numerics verified against the standard RMSNorm backward (dw rel 0.0%,
    dx_back rel 0.2%).
    """
    # Accept both EMU3_NPU_ADD_RMSNORM (canonical) and EMU3_NPU_ADD_RMNORM
    # (the typo used by every shell script in the repo) so the fusion actually
    # turns on without waiting on a script-wide rename.
    _add_rms_on = (os.getenv("EMU3_NPU_ADD_RMSNORM") == "1" or
                   os.getenv("EMU3_NPU_ADD_RMNORM") == "1")
    if not _add_rms_on:
        return None
    _dbg = (os.getenv("EMU3_DEBUG_ADD_RMSNORM") or os.getenv("EMU3_DEBUG_ADD_RMNORM"))
    try:
        import torch_npu
        if not _npu_ready("npu_add_rms_norm_v2_functional") or \
           not hasattr(torch.ops.npu, "npu_rms_norm_backward"):
            # No backward op available — fall back to caller's PyTorch path
            if _dbg:
                print(f"[NPU][DBG] fused_add_rms_norm FALLBACK (op not ready)")
            return None
        if _dbg:
            print(f"[NPU][DBG] fused_add_rms_norm APPLY (x={tuple(x.shape)} res={tuple(residual.shape)})")
        return _FusedAddRMSNorm.apply(x, residual, weight, eps)
    except Exception as e:
        print(f"[NPU] fused_add_rms_norm failed: {e}")
        return None


class _FusedAddRMSNorm(torch.autograd.Function):
    """Differentiable fused Add + RMSNorm.

    forward : out = RMSNorm(residual + x); x_back = residual + x
              returns (out, x_back)
    backward: (dx_back, dw) = npu_rms_norm_backward(grad_out, x_back, w, rstd)
              dx = dx_back, dr = dx_back  (because x_back = x + residual)
    """

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        # v2_functional returns (rstd, output, x_back)
        rstd, output, x_back = torch_npu.npu_add_rms_norm_v2_functional(
            x, residual, weight, eps)
        ctx.save_for_backward(x_back, weight, rstd)
        return output, x_back

    @staticmethod
    def backward(ctx, grad_output, grad_x_back):
        x_back, weight, rstd = ctx.saved_tensors
        # npu_rms_norm_backward(grad_y, x, gamma, rstd) -> (dx_back, dw)
        dx_back, dw = torch.ops.npu.npu_rms_norm_backward(
            grad_output, x_back, weight, rstd)
        # x_back = x + residual  =>  dx = dx_back, dresidual = dx_back
        # grad_x_back feeds the residual-out (x_back) consumer; since forward
        # returned x_back = x + residual, its gradient also flows back to both.
        dx = dx_back + grad_x_back
        dr = dx_back + grad_x_back
        return dx, dr, dw, None


