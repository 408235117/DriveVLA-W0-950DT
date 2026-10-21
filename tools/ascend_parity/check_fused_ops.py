"""Parity checks for the optional Ascend fused ops.

Run on the NPU box from the repo root:

    python tools/ascend_parity/check_fused_ops.py

Compares each torch_npu op against the PyTorch reference used in the vendored
transformers code. Exits non-zero on failure. No test framework by design.
"""

import sys

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:
    torch_npu = None

TOL_BF16 = 1e-2
TOL_FP32 = 1e-3

failures = []


def check(name, out, ref, tol, dtype_note=""):
    diff = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().clamp_min(1e-6).item()
    rel = diff / scale
    ok = rel <= tol
    print(f"{'PASS' if ok else 'FAIL'} {name}{dtype_note}: maxdiff={diff:.3e} rel={rel:.3e} tol={tol}")
    if not ok:
        failures.append(name)


def rotate_half(t):
    x1, x2 = t.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def reference_rms_norm(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)))


def main():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        print("SKIP: torch_npu / NPU not available")
        return 0

    dev = "npu"
    torch.manual_seed(0)
    print(f"torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', '?')}")

    for n in ("npu_rms_norm", "npu_swiglu", "npu_rotary_mul", "npu_gelu", "npu_fusion_attention"):
        print("available", n, hasattr(torch_npu, n))

    # --- RMSNorm ---
    x = torch.randn(24, 421, 3584, device=dev, dtype=torch.bfloat16)
    w = torch.randn(3584, device=dev, dtype=torch.bfloat16)
    ref = reference_rms_norm(x, w, 1e-6)
    out = torch_npu.npu_rms_norm(x, w, epsilon=1e-6)[0]
    check("rms_norm bf16", out, ref, TOL_BF16)

    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    torch_npu.npu_rms_norm(x2, w2, epsilon=1e-6)[0].float().pow(2).mean().backward()
    finite = bool(torch.isfinite(x2.grad).all()) and bool(torch.isfinite(w2.grad).all())
    print(f"{'PASS' if finite else 'FAIL'} rms_norm backward finite: "
          f"dx_max={float(x2.grad.abs().max()):.3e} dw_max={float(w2.grad.abs().max()):.3e}")
    if not finite:
        failures.append("rms_norm backward")

    # --- SwiGLU (tensor, dim=-1 convention) ---
    gate = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    up = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    ref = F.silu(gate.float()) * up.float()
    out = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)
    check("swiglu bf16", out, ref, TOL_BF16)

    # --- mRoPE / rotate_half, full-width cos/sin ---
    B, H, L, D = 24, 28, 421, 128
    q = torch.randn(B, H, L, D, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(B, 1, L, D, device=dev, dtype=torch.bfloat16)
    sin = torch.randn(B, 1, L, D, device=dev, dtype=torch.bfloat16)
    ref = q.float() * cos.float() + rotate_half(q.float()) * sin.float()
    out = torch_npu.npu_rotary_mul(q, cos, sin)
    check("rotary_mul bf16", out, ref, TOL_BF16)

    # --- vision RoPE shapes ---
    S, HV, DV = 1024, 16, 64
    qv = torch.randn(S, HV, DV, device=dev, dtype=torch.bfloat16)
    cosv = torch.randn(S, 1, DV, device=dev, dtype=torch.bfloat16)
    sinv = torch.randn(S, 1, DV, device=dev, dtype=torch.bfloat16)
    refv = qv.float() * cosv.float() + rotate_half(qv.float()) * sinv.float()
    outv = torch_npu.npu_rotary_mul(qv, cosv, sinv)
    check("rotary_mul vision bf16", outv, refv, TOL_BF16)

    # --- exact GELU ---
    xg = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    refg = F.gelu(xg.float(), approximate="none")
    outg = torch_npu.npu_gelu(xg, approximate="none")
    check("gelu exact bf16", outg, refg, TOL_BF16)

    # --- FA2 vs SDPA (causal, GQA 28/4, BSND) ---
    Bq, Sq, Hq, Hkv, Dq = 2, 512, 28, 4, 128
    qa = torch.randn(Bq, Sq, Hq, Dq, device=dev, dtype=torch.bfloat16)
    ka = torch.randn(Bq, Sq, Hkv, Dq, device=dev, dtype=torch.bfloat16)
    va = torch.randn(Bq, Sq, Hkv, Dq, device=dev, dtype=torch.bfloat16)
    mask = torch.triu(torch.ones(Sq, Sq, device=dev, dtype=torch.bool), diagonal=1)
    fa = torch_npu.npu_fusion_attention(
        qa, ka, va, Hq, "BSND", keep_prob=1.0, scale=Dq ** -0.5,
        atten_mask=mask, sparse_mode=3,
    )[0]
    rep = Hq // Hkv
    k_r = ka.transpose(1, 2).repeat_interleave(rep, dim=1)
    v_r = va.transpose(1, 2).repeat_interleave(rep, dim=1)
    sdpa = F.scaled_dot_product_attention(
        qa.transpose(1, 2).float(), k_r.float(), v_r.float(), is_causal=True
    ).transpose(1, 2)
    check("fa2 causal vs sdpa bf16", fa, sdpa, TOL_BF16)

    print()
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
