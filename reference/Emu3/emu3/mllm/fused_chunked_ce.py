"""Fused chunked lm_head + cross-entropy custom autograd operator.

A drop-in replacement for ``chunked_lm_head_cross_entropy`` that cuts the number
of lm_head matmuls per training step from ~3x to ~1.5x by:

  * Forward: single-pass *online* logsumexp over the vocab dimension
    (running-max + rescaled accumulator), instead of the original two-pass
    "find-global-max-then-lse" scan. The first pass computes ``running_max``
    purely for numerical stability of the exp(); its gradient w.r.t. the loss
    is mathematically zero (max is a stop-gradient for the lse), so dropping it
    is numerically faithful *and* removes ~1/3 of the lm_head matmuls.
  * Backward: a hand-written ``backward`` that recomputes the per-chunk logits
    exactly once (standard Liger-Kernel pattern), instead of letting autograd
    recompute them *twice* (once for the max pass, once for the lse pass of the
    two-pass forward) — i.e. autograd would otherwise redo the whole forward
    scan in backward.

Net: vocab is chunked (default 8192); peak intermediate stays [N, chunk]. Only
``hidden_states`` (kept in its original 3-D [B,S,H] form, ~335 MiB at
bs32/seq1280/H4096/bf16) + the scalar ``lse [N]`` are saved for backward — no
full [N,V] logits (~13 GiB), and ``lm_head_weight`` is saved by reference only
(it is a model parameter already resident in HBM, so this costs nothing extra).

Shape handling: ``hidden_states`` is passed through UNMODIFIED (no flatten, no
reshape) — it stays 3-D [B,S,H] (or 2-D [N,H]) exactly as received, so the
backward gradient returned to its producer (fused RMSNorm) has the identical
shape/contiguity, keeping the autograd edge clean under async NPU streams +
ZeRO. Only the per-chunk logits are reshaped to 2-D [N, c] for vocab-dim ops.

HBM regime: ~37 GiB free (58.79/96 GB used at bs32), so saving the small
``hidden_states`` activation is strictly better than the reentrant-checkpoint
recompute trade (which saves HBM by burning compute — the wrong trade in a
compute-bound step).

Env flag ``EMU3_FUSED_CE=1`` enables it at the call site; otherwise the original
``chunked_lm_head_cross_entropy`` path is used (backward compatible, bit-faithful
to the baseline).

Env flag ``EMU3_CE_ONLINE=1`` (default OFF; the default path above stays
untouched) additionally selects a fused-kernel rewrite of the chunk loop
*inside* this operator — same math, fewer/larger elementwise kernels:
``torch.mm(out=)`` writes the logits directly into contiguous per-chunk
buffers, ``torch.logsumexp`` + ``torch.logaddexp`` build the lse, and the
backward uses ``torch.softmax(dtype=fp32)`` (rescaled to the global softmax)
+ one in-place row multiply. See ``OnlineChunkedLinearCrossEntropy``. This
path implies the save-logits HBM trade (the buffers ARE the saved logits),
so ``EMU3_FUSED_CE_SAVELOGITS`` is ignored while it is on.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _resolve_chunk_size(V: int, chunk_size: int) -> int:
    if chunk_size <= 0 or chunk_size > V:
        return V
    return chunk_size


class FusedChunkedLinearCrossEntropy(torch.autograd.Function):
    """lm_head (F.linear over vocab chunks) + cross-entropy, fused, 1-pass.

    Forward computes ``loss = mean( logsumexp(logits[label]) - logits[label] )``
    over valid (non-ignore) tokens, without ever materialising the full
    ``[N, V]`` logits tensor.

    Backward computes ``grad_hidden`` (and ``grad_weight`` if requested) via a
    single recompute of the per-chunk logits (Liger-Kernel pattern), avoiding
    the double-recompute that autograd performs on the two-pass reference impl.
    """

    @staticmethod
    def forward(ctx, hidden_states, lm_head_weight, labels,
                chunk_size=8192, ignore_index=-100, weight=None):
        # Keep hidden_states in its ORIGINAL shape (3-D or 2-D). Never flatten
        # it — the autograd edge back to the fused RMSNorm producer must receive
        # a gradient of identical shape/contiguity. F.linear handles 3-D natively.
        orig_ndim = hidden_states.ndim
        if orig_ndim == 3:
            B, S, H = hidden_states.shape
            N = B * S
        else:
            N, H = hidden_states.shape
        V = lm_head_weight.shape[0]
        device = hidden_states.device
        cs = _resolve_chunk_size(V, chunk_size)

        # HBM-for-compute trade (PDF 案例D3/D4 "显存预算即性能预算"): when
        # EMU3_FUSED_CE_SAVELOGITS=1, materialise the full [N,V] logits in
        # forward and reuse them in backward, eliminating the per-chunk F.linear
        # recompute (~0.28s, ~1x of the lm_head matmul — the backward half of
        # the 2x the fused-CE op now does). Costs [N,V] bf16 (~15GiB at
        # bs32/seq1280/v184622), within the ~44GiB free after fused CE
        # (peak ~67/96GiB). Numerically identical to the recompute path: the
        # saved logits are the SAME bf16 F.linear outputs, upcast to fp32 for
        # the softmax in both paths.
        save_logits = bool(int(os.environ.get("EMU3_FUSED_CE_SAVELOGITS", "0")))

        # --- Single-pass online logsumexp over vocab chunks ---
        # running_max[m] = max_v logits[m,v]
        # running_lse[m] = sum_v exp(logits[m,v] - running_max[m])
        # final lse[m]    = log(running_lse[m]) + running_max[m]
        # Accumulate in fp32 for numerical stability (the reference impl
        # accumulates in bf16 — fp32 here is strictly more accurate, and HBM
        # headroom makes the 2x lse tensor size a non-issue).
        running_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        running_lse = torch.zeros((N,), device=device, dtype=torch.float32)
        label_logit = torch.zeros((N,), device=device, dtype=torch.float32)
        # When save_logits is on, allocate the full [N,V] bf16 logits buffer
        # up-front and fill it chunk-by-chunk; backward reads it back and skips
        # the F.linear recompute. This is the only divergence from the recompute
        # path — everything else is identical.
        saved_logits = torch.empty((N, V), device=device, dtype=hidden_states.dtype) if save_logits else None

        for i in range(0, V, cs):
            chunk_w = lm_head_weight[i:i + cs]                       # [c, H]
            chunk_l = F.linear(hidden_states, chunk_w)               # [B,S,c] or [N,c]
            # flatten ONLY the logits (contiguous F.linear output, free view)
            # for vocab-dim reductions — hidden_states is left untouched.
            chunk_l = chunk_l.reshape(N, -1)                         # [N, c]
            if saved_logits is not None:
                saved_logits[:, i:i + cs] = chunk_l                  # keep for backward
            chunk_l_f = chunk_l.float()                               # [N, c] fp32

            chunk_max = chunk_l_f.amax(dim=-1)                        # [N]
            new_max = torch.maximum(running_max, chunk_max)
            # rescale running accumulator to the new max, then add this chunk
            running_lse = (running_lse * torch.exp(running_max - new_max)
                          + torch.exp(chunk_l_f - new_max.unsqueeze(-1)).sum(dim=-1))
            running_max = new_max

            # gather the logit at each token's label position, fully vectorised
            # (no .any() host sync, no Python control-flow branch per chunk).
            local_idx = labels - i
            valid = (local_idx >= 0) & (local_idx < cs)
            safe_idx = local_idx.clamp(min=0, max=cs - 1)
            gathered = chunk_l_f.gather(1, safe_idx.unsqueeze(-1)).squeeze(-1)
            label_logit = torch.where(valid, gathered, label_logit)

        lse = running_lse.log() + running_max                          # [N] logsumexp
        nll = lse - label_logit                                        # [N]

        if weight is not None:
            w = weight[labels.clamp(min=0)]
            w = torch.where(labels == ignore_index, torch.zeros_like(w), w)
            nll = nll * w

        mask = labels != ignore_index
        n_valid = mask.sum()
        # Static-shape masked sum: the pre-fix boolean indexing lowered to
        # aclnnNonzeroV2, whose dynamic output size forces a host sync —
        # prof_54 api_statistic shows one call blocking the host for 1.69s
        # (device kernel: 8.9us), draining the queue and opening a ~30ms
        # 8-rank bubble between the forward tail and backward entry.
        # torch.where computes the identical sum with static shapes, and —
        # unlike nll*mask — cannot propagate a non-finite nll at an ignored
        # position into the loss.
        loss = torch.where(mask, nll, torch.zeros_like(nll)).sum() / n_valid.clamp(min=1).to(nll.dtype)

        # --- save for backward ---
        # Save hidden_states in its ORIGINAL shape (3-D or 2-D) + lse (cheap
        # [N] fp32). We do NOT save logits (would be [N,V] ~13 GiB).
        # lm_head_weight is saved by reference (it is a parameter; not copied).
        # Backward recomputes per-chunk logits.
        ctx.save_for_backward(hidden_states, lm_head_weight, labels, lse.detach(), saved_logits)
        ctx.chunk_size = cs
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid.detach()
        ctx.V = V
        ctx.orig_ndim = orig_ndim
        ctx.needs_weight_grad = lm_head_weight.requires_grad
        ctx.save_weight = weight.detach() if weight is not None else None
        ctx.mask = mask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, lm_head_weight, labels, lse, saved_logits = ctx.saved_tensors
        cs = ctx.chunk_size
        ignore_index = ctx.ignore_index
        # n_valid stays a 0-dim tensor: pulling it to host (int(...item()))
        # is a sync of the same family as the nonzero removed in forward.
        # The division below broadcasts; fp32/int64 promotion reproduces the
        # original scalar division exactly (n_valid < 2^24 is exact in fp32).
        n_valid = ctx.n_valid
        V = ctx.V
        mask = ctx.mask
        weight = ctx.save_weight
        device = hidden_states.device
        dtype = hidden_states.dtype
        orig_ndim = ctx.orig_ndim

        if orig_ndim == 3:
            B, S, H = hidden_states.shape
            N = B * S
        else:
            N, H = hidden_states.shape

        # grad w.r.t. each valid token's loss is grad_output / n_valid (mean
        # over valid tokens). Ignored tokens contribute zero gradient.
        go = grad_output / n_valid.clamp(min=1)                            # 0-dim tensor, no host sync

        # grad_hidden keeps the EXACT shape of hidden_states (3-D or 2-D) so the
        # autograd edge back to the fused RMSNorm producer is shape-faithful.
        grad_hidden = torch.zeros_like(hidden_states) if hidden_states.requires_grad else None
        grad_weight = torch.zeros_like(lm_head_weight) if ctx.needs_weight_grad else None

        # softmax = exp(logits - lse). Recompute logits per chunk, build
        # grad_logits = (softmax - onehot) * scale, then accumulate
        #   grad_hidden += grad_logits @ W_chunk   (via F.linear)
        #   grad_W_chunk += grad_logits^T @ hidden
        # NOTE: everything here is out-of-place (functional). In-place ops on
        # autograd-graph tensors break NPU CANN's stream/version accounting under
        # async exec + fused RMSNorm (manifests as aclnnRmsNormGrad "dy==nullptr").
        # hidden_2d is a view of hidden_states (3-D) used only for the grad_W
        # matmul, never modifying hidden_states itself.
        hidden_2d = hidden_states.reshape(N, H)
        arange = torch.arange(N, device=device)
        for i in range(0, V, cs):
            chunk_w = lm_head_weight[i:i + cs]                        # [c, H] (cheap slice; needed for the grad matmuls below in both paths)
            if saved_logits is not None:
                # save-logits path: reuse the forward logits, NO F.linear recompute.
                # This is the whole point of the HBM-for-compute trade — skips the
                # ~0.28s per-chunk recompute that the recompute path pays below.
                chunk_l = saved_logits[:, i:i + cs]                  # [N, c]
            else:
                chunk_l = F.linear(hidden_states, chunk_w)            # [B,S,c] or [N,c]
                chunk_l = chunk_l.reshape(N, -1)                     # [N, c]
            chunk_l_f = chunk_l.float()
            softmax = torch.exp(chunk_l_f - lse.unsqueeze(-1))       # [N, c] (new tensor)

            # subtract one-hot at the label position, OUT-OF-PLACE: build a
            # delta tensor (zeros everywhere except -1 at in-chunk label rows)
            # and add it to softmax. Avoids in-place scatter on the autograd
            # tensor that fused-RMSNorm backward is sensitive to.
            local_idx = labels - i
            valid = (local_idx >= 0) & (local_idx < cs)
            safe_idx = local_idx.clamp(min=0, max=cs - 1)
            delta = torch.zeros_like(softmax)
            delta[arange, safe_idx] = -valid.to(delta.dtype)
            softmax = softmax + delta

            # per-token scale: grad_output/n_valid * class_weight * ignore-mask
            if weight is not None:
                w_tok = weight[labels.clamp(min=0)]
                w_tok = torch.where(labels == ignore_index, torch.zeros_like(w_tok), w_tok)
                scale = go * w_tok.unsqueeze(-1) * mask.unsqueeze(-1).to(softmax.dtype)
            else:
                scale = go * mask.unsqueeze(-1).to(softmax.dtype)
            grad_logits = (softmax * scale).to(dtype)                # [N, c]

            # grad_hidden += grad_logits @ W_chunk  ->  [N, H]
            # F.linear(grad_logits, W_chunk.t()) = grad_logits @ W_chunk.
            # In-place add_ onto the pre-allocated grad_hidden keeps every chunk's
            # accumulation on the SAME tensor/stream — avoiding the per-chunk
            # tensor rebuilds that confused ZeRO's copy_stream bookkeeping
            # (aclnnRmsNormGrad "dy==nullptr" was triggered by the stream churn
            # of `grad_hidden = grad_hidden + ...` across 23 chunks).
            if grad_hidden is not None:
                gh_chunk = F.linear(grad_logits, chunk_w.t())        # [N, H]
                gh_chunk = gh_chunk.reshape_as(grad_hidden)
                grad_hidden.add_(gh_chunk)
            # grad_W_chunk += grad_logits^T @ hidden  ->  [c, H]
            if grad_weight is not None:
                grad_weight[i:i + cs] = grad_logits.t() @ hidden_2d

        # 6 return values correspond to the 6 forward inputs:
        # (hidden_states, lm_head_weight, labels, chunk_size, ignore_index, weight)
        return (grad_hidden, grad_weight, None, None, None, None)


class OnlineChunkedLinearCrossEntropy(torch.autograd.Function):
    """Fused-kernel rewrite of the chunk loop — enabled by EMU3_CE_ONLINE=1.

    Same math and same HBM budget as FusedChunkedLinearCrossEntropy with
    save-logits on (the per-chunk contiguous buffers ARE the saved logits,
    ~[N,V] bf16 total), but the per-chunk elementwise chains are replaced by
    fused torch ops (probe-verified on 950DT, N=40928 c=8192):

      forward : ``torch.mm(out=)`` writes bf16 logits straight into a
                contiguous [N, c] buffer (no [N, V] strided ViewCopy, no
                3-D non-contiguous F.linear); lse = per-chunk
                ``logsumexp(fp32)`` merged with ``logaddexp`` (seeded with
                -inf, exact at the boundary); label logits are gathered on
                the bf16 buffer (a select — identical values to cast-first).
      backward: reads the contiguous buffers directly (no strided Slice),
                builds the GLOBAL softmax as ``torch.softmax(buf, dtype=fp32)``
                rescaled by ``exp(chunk_lse - lse)`` — the rescale factor is
                folded into the single in-place row multiply — then fixes the
                label column with one index_put (invalid rows write 0 over an
                already-zero row).

    The per-chunk lse (the logaddexp merge inputs) is saved so the backward
    rescale is exact. Probe-estimated effect vs the save-logits path: fwd Vec
    ~86 -> ~65ms (logsumexp replaces the amax/sub/exp/reducesum rescale
    chain; ViewCopy and Matmul_Slice disappear), bwd Vec ~141 -> ~68ms
    (softmax + mul_ + one cast replace sub/exp/zeros_like/add/mul/cast and
    the strided Slice), i.e. ~90-100ms/step at bs32/seq1280/v184622.
    """

    @staticmethod
    def forward(ctx, hidden_states, lm_head_weight, labels,
                chunk_size=8192, ignore_index=-100, weight=None):
        orig_ndim = hidden_states.ndim
        if orig_ndim == 3:
            B, S, H = hidden_states.shape
            N = B * S
        else:
            N, H = hidden_states.shape
        V = lm_head_weight.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        cs = _resolve_chunk_size(V, chunk_size)

        # ONE contiguous [N,H] copy up-front (~0.2ms when the training input
        # is the non-contiguous [B, S-1, H] slice): every chunk matmul then
        # runs on 2-D contiguous operands and writes with out=.
        hidden_2d = hidden_states.reshape(N, H)

        running_lse = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        label_logit = torch.zeros((N,), device=device, dtype=torch.float32)

        bufs = []
        chunk_lses = []
        for i in range(0, V, cs):
            chunk_w = lm_head_weight[i:i + cs]                       # [c, H] view
            c = chunk_w.shape[0]
            # contiguous per-chunk buffer; mm writes the bf16 logits in place
            buf = torch.empty((N, c), device=device, dtype=dtype)
            torch.mm(hidden_2d, chunk_w.t(), out=buf)
            bufs.append(buf)

            chunk_lse = torch.logsumexp(buf.float(), dim=-1)         # [N] fp32
            chunk_lses.append(chunk_lse)
            running_lse = torch.logaddexp(running_lse, chunk_lse)    # log-domain merge

            # label gather on the bf16 buffer (select, then [N]-level cast)
            local_idx = labels - i
            valid = (local_idx >= 0) & (local_idx < c)
            safe_idx = local_idx.clamp(min=0, max=c - 1)
            gathered = buf.gather(1, safe_idx.unsqueeze(-1)).squeeze(-1).float()
            label_logit = torch.where(valid, gathered, label_logit)

        lse = running_lse                                            # already log-domain
        nll = lse - label_logit

        if weight is not None:
            w = weight[labels.clamp(min=0)]
            w = torch.where(labels == ignore_index, torch.zeros_like(w), w)
            nll = nll * w

        mask = labels != ignore_index
        n_valid = mask.sum()
        # Static-shape masked sum — same NonzeroV2 host-sync fix as above.
        loss = torch.where(mask, nll, torch.zeros_like(nll)).sum() / n_valid.clamp(min=1).to(nll.dtype)

        # save: original-shape hidden (clean autograd edge) + the contiguous
        # 2-D copy + the per-chunk buffers (the saved logits) + stacked lse.
        ctx.save_for_backward(hidden_states, lm_head_weight, labels, lse.detach(),
                              hidden_2d, *bufs)
        ctx.chunk_lse_all = torch.stack(chunk_lses)                  # [K, N] fp32
        ctx.chunk_size = cs
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid.detach()
        ctx.V = V
        ctx.orig_ndim = orig_ndim
        ctx.needs_weight_grad = lm_head_weight.requires_grad
        ctx.save_weight = weight.detach() if weight is not None else None
        ctx.mask = mask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, lm_head_weight, labels, lse, hidden_2d, *bufs = ctx.saved_tensors
        cs = ctx.chunk_size
        ignore_index = ctx.ignore_index
        # n_valid stays a 0-dim tensor on device — same host-sync discipline
        # as FusedChunkedLinearCrossEntropy.backward.
        n_valid = ctx.n_valid
        mask = ctx.mask
        weight = ctx.save_weight
        chunk_lse_all = ctx.chunk_lse_all
        device = hidden_states.device
        dtype = hidden_states.dtype
        orig_ndim = ctx.orig_ndim

        if orig_ndim == 3:
            B, S, H = hidden_states.shape
            N = B * S
        else:
            N, H = hidden_states.shape

        go = grad_output / n_valid.clamp(min=1)                      # 0-dim, no host sync

        grad_hidden = torch.zeros_like(hidden_states) if hidden_states.requires_grad else None
        grad_weight = torch.zeros_like(lm_head_weight) if ctx.needs_weight_grad else None

        arange = torch.arange(N, device=device)
        # per-token scale row, built ONCE (the old loop rebuilt it per chunk):
        # grad_output/n_valid * class_weight * ignore-mask
        if weight is not None:
            w_tok = weight[labels.clamp(min=0)]
            w_tok = torch.where(labels == ignore_index, torch.zeros_like(w_tok), w_tok)
            scale_row = go * w_tok * mask.to(torch.float32)
        else:
            scale_row = go * mask.to(torch.float32)
        scale_row = scale_row.unsqueeze(-1)                          # [N, 1] fp32

        for k, buf in enumerate(bufs):
            i = k * cs
            c = buf.shape[1]
            chunk_w = lm_head_weight[i:i + c]                        # [c, H] view

            local_idx = labels - i
            valid = (local_idx >= 0) & (local_idx < c)
            safe_idx = local_idx.clamp(min=0, max=c - 1)

            # chunk-local softmax in fp32, then rescale to the GLOBAL softmax:
            # p[m, i+j] = sm[m, j] * exp(chunk_lse[k] - lse)
            sm = torch.softmax(buf, dim=-1, dtype=torch.float32)     # [N, c] new tensor
            p_lab = sm.gather(1, safe_idx.unsqueeze(-1)).squeeze(-1) # BEFORE the mul_
            delta = torch.exp(chunk_lse_all[k] - lse)                # [N]

            # ONE in-place row multiply folds both the per-token scale and
            # the global-softmax rescale. In-place is safe here: sm is a
            # fresh tensor created inside backward (no autograd history) —
            # the same discipline as the delta index_put in the original
            # backward above.
            g = sm.mul_(scale_row * delta.unsqueeze(-1))
            # label-column fix: (p_global[label] - 1) * scale; invalid rows
            # write 0 over an already-zero row (probe: maxdiff 5.96e-08).
            g[arange, safe_idx] = (p_lab * delta - valid.to(torch.float32)) * scale_row.squeeze(-1)
            grad_logits = g.to(dtype)                                # [N, c] bf16

            # grad_hidden += grad_logits @ W_chunk (same in-place accumulation
            # onto one pre-allocated tensor — keeps ZeRO's copy_stream happy)
            if grad_hidden is not None:
                gh_chunk = torch.mm(grad_logits, chunk_w)            # [N, H]
                grad_hidden.add_(gh_chunk.reshape_as(grad_hidden))
            if grad_weight is not None:
                grad_weight[i:i + c] = grad_logits.t() @ hidden_2d

        # 6 return values correspond to the 6 forward inputs:
        # (hidden_states, lm_head_weight, labels, chunk_size, ignore_index, weight)
        return (grad_hidden, grad_weight, None, None, None, None)


def fused_chunked_linear_cross_entropy(hidden_states, lm_head_weight, labels,
                                       chunk_size=8192, ignore_index=-100, weight=None):
    """Public entry: FusedChunkedLinearCrossEntropy.apply(...).

    EMU3_CE_ONLINE=1 selects the fused-kernel rewrite
    OnlineChunkedLinearCrossEntropy (implies the save-logits HBM trade;
    EMU3_FUSED_CE_SAVELOGITS is ignored on that path). Default OFF.
    """
    if bool(int(os.environ.get("EMU3_CE_ONLINE", "0"))):
        return OnlineChunkedLinearCrossEntropy.apply(
            hidden_states, lm_head_weight, labels, chunk_size, ignore_index, weight)
    return FusedChunkedLinearCrossEntropy.apply(
        hidden_states, lm_head_weight, labels, chunk_size, ignore_index, weight)
