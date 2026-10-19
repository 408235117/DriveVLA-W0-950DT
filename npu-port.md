# Qwen2.5-VL NPU Port — Work Log

**Base commit:** `db45a6f7f2d6e966b6b94166a33dcf8e1014804e`
**Branch:** `qwen-npu-port`
**Scope:** train the Qwen2.5-VL (ROSS) VLA backbone on Ascend NPU, starting with a
synthetic-data throughput run.

This document records every change made on the branch since the base commit, the
reason for it, and what is still outstanding.

---

## Step 0 — Baseline audit (no code change)

Entry point: [`utils/train_qwen_vla.py`](utils/train_qwen_vla.py).

Findings that drove the subsequent changes:

| Finding | Evidence |
|---|---|
| Backbone is **Qwen2.5-VL-7B**, not 3B | HF `liyingyan/DriveVLA-W0/Qwen_PDMS_85.3/config.json`: `hidden_size=3584`, `num_hidden_layers=28` (3B is `2048`/`36`). The 3B values in `argument.py` and the deleted launcher are stale defaults. |
| The model needs the **vendored transformers 4.56.0** | `reference/transformers/src/transformers/__init__.py:21`. `requirements.txt` pins `4.44.0`, which has no `qwen2_5_vl` module. The Qwen model/config/processor files are byte-identical to upstream `v4.56.0`; the only differences are the ROSS additions, a one-line import in `trainer.py`, and a ROSS reference in the Qwen `__init__.py`. |
| ROSS hardcodes a **10×18 = 180-token** image grid | `modeling_qwen2_5_vl_ross.py:68` (`n_patches=180`) and `:359` (`rearrange(... h=10, w=18)`). The default `max_pixels=28*28*576` produces 576 tokens for 16:9 images and crashes. |
| ROSS asserts **exactly 2 image/action segments** | `modeling_qwen2_5_vl_ross.py:335-337`. `use_previous_actions=true` with `cur_frame_idx>0` adds two more action segments and trips the assert. The old launcher avoided this with `cur_frame_idx=0`, which silently suppressed history. |
| Per-device batch must be **≥ 2** | `image_hidden[:, 0].squeeze()` at `:339` removes the batch dim when batch size is 1. |
| Training requires **Stable Diffusion 1.5** | UNet at `denoiser_sd.py:41`, VAE at `modeling_qwen2_5_vl_ross.py:71-81`, scheduler at `denoiser_sd.py:55`; `--sd_model_path` must end in `/unet` because paths are derived with `.replace("/unet", ...)`. |
| `decord` has **no aarch64 wheel** | Imported unconditionally by `qwenvl/dataset/data_qwen_vla.py` and `rope2d.py`, but `VideoReader` is never used on the VLA path. |

Image path resolution (`data_qwen_vla.py:391-424`): the pickle's
`.../<log>/<frame>.npy` is rewritten to
`<data_root>/<log>/CAM_F0/<frame>.jpg`.

---

## Step 1 — `1701619` Add Qwen2.5-VL VLA training script and requirements file

### 1a. `requirements_qwen.txt` (new)

Added a Qwen-specific environment file because `requirements.txt` is for the
Emu3 path and its `transformers==4.44.0` is incompatible.

Final contents (evolved over commits `1701619`, `2088d65`, `877dfd8`):

- `-e ./reference/transformers` — editable 4.56.0 so `import transformers`
  resolves to the vendored patched tree and port edits are live.
- NPU torch stack: `torch==2.7.1`, `torchvision==0.22.1`,
  `torchaudio==2.7.1`, `torch_npu==2.7.1.post4`.
- `deepspeed==0.17.6`, `accelerate`, `tensorboard`, `numpy`, `pillow`,
  `pyyaml`, `tqdm`, `safetensors`.
- `diffusers`, `einops==0.8.1`, `scipy`.
- `decord` removed (see Step 4); `flash-attn` left commented because it is
  CUDA-only.

Install order is documented in the file header: CANN 8.5.1 must be installed
and `set_env.sh` sourced before `torch_npu`.

### 1b. `scripts/scripts_train/train_qwen_vla_navsim.sh` (new)

The upstream launcher for this path had been deleted in `929171a`. It was
recovered for reference (`git show eb82f31:scripts/scripts_train/train_qwen_vla_navsim.sh`)
but not copied verbatim, because it passes flags that no longer exist and uses
stale defaults.

Differences from the deleted launcher:

| Old launcher | New launcher |
|---|---|
| `--model_name_or_path ...3B...` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| passes `--ross_loss_weight`, `--ross_grad_clip` | omitted — removed from `ModelArguments` at HEAD |
| `cur_frame_idx=0` | `3` (matches the pickle window `-3..+8`) |
| no `max_pixels` | `150528` → 180 tokens for 16:9 |
| no SD path validation | resolves `<parent>/unet`, validates `vae/` + `scheduler/` |
| no env preflight | checks `transformers` has Qwen2.5-VL; warns if `flash_attn` missing |
| `use_previous_actions=true` | `false` (Step 2) |

Features of the new launcher:

- `--device auto|cuda|npu`, forwarded via the `DEVICE` environment variable.
- NPU env: `INF_NAN_MODE_ENABLE=1`, `HCCL_DETERMINISTIC=TRUE`,
  `ASCEND_RT_VISIBLE_DEVICES`; CUDA: `NCCL_DETERMINISTIC`,
  `CUDA_DEVICE_MAX_CONNECTIONS`, `PYTORCH_CUDA_ALLOC_CONF`.
- `PYTHONPATH` includes the repo root (for `reference.*` imports) and
  `reference/Qwen2.5-VL/qwen-vl-finetune`.
- Rejects `fa2` on NPU; `sdpa` is the default.
- Zero-stage shortcuts `--zero_stage 2|3` → `scripts/sft/zero{2,3}_offload.json`.
- Writes the full invocation to `<run_dir>/startup_command.txt`.
- Resume is automatic: `train_qwen_vla.py` detects `checkpoint-*` in the run dir.

---

## Step 2 — `f9a61e3` Update training script to disable previous actions and add synthetic dataset generation script

### 2a. `use_previous_actions` default changed to `false`

Reason: the ROSS forward asserts `action_hidden.shape[1] == 2`
(`modeling_qwen2_5_vl_ross.py:336`). `build_action_segments` adds four
segments when history is enabled and `cur_frame_idx > 0` (pre-user,
pre-assistant, cur-user, cur-assistant), so the assert fails. With history
disabled, exactly two assistant segments remain.

Comment added next to the default in the launcher.

### 2b. `tools/synth/synth_qwen_vla_data.py` (new)

Purpose: measure throughput without downloading the 1.24 TB NavSim camera set.

Generated schema matches `data_qwen_vla.py`:

- One scene = one training sample.
- `text` / `pre_1s_text`: `list[12]` strings.
- `image` / `pre_1s_image`: `list[12]` `.npy`-style paths.
- `action` / `pre_1s_action`: `[12, 3]` float32 in `[-1, 1]`.
- Images are **504×280**, i.e. exactly 10×18 = 180 Qwen vision tokens.
- Files resolve to `<data_root>/<log>/CAM_F0/<frame>.jpg`.

Generated artifact (not committed; `logs/` is gitignored):

```
logs/synth_qwen_data/
├── synth_navsim.pkl        # 256 scenes
└── images/<log>/CAM_F0/*.jpg  # 12 per scene = 3,072 files, ~12 MB
```

Verified: 256 samples, correct keys, first image resolves, resize grid = 180.

Throughput command:

```bash
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
    --data_path logs/synth_qwen_data/synth_navsim.pkl \
    --data_root logs/synth_qwen_data/images \
    --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5 \
    --model_name_or_path /mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
    --device npu --ngpus 8 --batch_size 2 \
    --max_steps 20 --logging_steps 1 \
    --save_steps 100000 --save_total_limit 1
```

Note: on the 8-rank NPU run the remote pkl contained 4,096 scenes and the
per-rank dataloader length was 256 (= 4096 / 8 / batch 2), so the ranks were
sharded. The local generator default is 256 scenes; regenerate with
`--num_scenes 4096` if you want the same sample count.

---

## Step 3 — `2088d65` Update requirements_qwen.txt to clarify installation instructions for torch stack on Ascend NPU

Added the NPU torch stack directly to the requirements file, per request:

```text
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
torch_npu==2.7.1.post4
```

Header updated with:

- CANN 8.5.1 is system-level; source `/usr/local/Ascend/ascend-toolkit/set_env.sh`.
- CUDA users must replace these four lines with torch 2.4.0 / torchvision
  0.19.0 / torchaudio 2.4.0 and drop `torch_npu`.
- DeepSpeed's ZeRO CPU offload JIT-builds CPUAdam, so a C++ toolchain is needed.

Verified on PyPI: `torch_npu==2.7.1.post4` ships cp39–cp313 wheels for
`manylinux_2_28_{x86_64,aarch64}`; `torch==2.7.1` has both architectures.

---

## Step 4 — `877dfd8` Refactor decord import to handle ImportError and update requirements to clarify its optional status

Triggered by the remote install failing:

```
ERROR: No matching distribution found for decord
```

`decord` has no aarch64 wheel, and pip aborts the whole transaction on one
unresolvable requirement — so `transformers` was never installed.

Changes:

- [`data_qwen_vla.py:25-28`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py#L25-L28):
  guarded import; `VideoReader = None` on `ImportError`.
- [`rope2d.py:19-22`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py#L19-L22):
  same guard.
- `requirements_qwen.txt`: `decord` removed, with a note.

`VideoReader` is not referenced elsewhere in either file, so the guard is safe
for the image-only VLA path.

---

## Step 5 — NPU runtime fixes (applied, not yet committed)

Triggered by the first real launch on the NPU host:

```
ModuleNotFoundError: No module named 'flash_attn'
  at qwenvl/train/trainer.py:7
```

Four fixes were applied so the training path can start on NPU and reach a
forward/backward pass:

| # | File | Change |
|---|---|---|
| 1 | [`qwenvl/train/trainer.py:7`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py#L7) | `flash_attn` import wrapped in `try/except ImportError` (`flash_attn_varlen_func = None`); `_flash_attention_forward` now raises a clear error only if `fa2` is actually used. `sdpa` runs without flash-attn. |
| 2 | [`modeling_ross/denoiser_sd.py:22-32`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L22-L32) | `negative_prompt_path` default changed from the author's `/mnt/vdb1/...` path to `None`, resolved relative to `__file__` (`negative_prompt_sd15.pt` next to the module). Added `import os`. |
| 3 | [`modeling_ross/denoiser_sd.py:81,195`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L81) | Both `torch.load(...)` calls now pass `map_location="cpu"` so the embeddings load on a machine without CUDA. |
| 4 | [`modeling_qwen2_5_vl_ross.py:355`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L355) | `torch.amp.autocast('cuda', ...)` → `torch.amp.autocast(image_hidden.device.type, ...)` (`npu` on Ascend, `cuda` on GPU). |
| 5 | [`utils/train_qwen_vla.py:280`](utils/train_qwen_vla.py#L280) | `torch.cuda.synchronize()` in `safe_save_model_for_hf_trainer` → existing `device_synchronize()` helper. |

Verification: these fixes were exercised by the successful 8-NPU run in Step 7.

---

## Step 6 — Missing `transformers.data` package (committed as `67292e7`)

The next launch got past the flash-attn guard and failed importing the Trainer:

```
ModuleNotFoundError: No module named 'transformers.data'
  at reference/transformers/src/transformers/trainer.py:59
```

Root cause: the vendored transformers checkout is a subset and omits the
upstream `data/` package entirely (`git log --all -- .../transformers/data`
returns nothing). `trainer.py:59` imports `DataCollator`,
`DataCollatorWithPadding`, and `default_data_collator` from it.

Fix:

- Added [`data/data_collator.py`](reference/transformers/src/transformers/data/data_collator.py)
  verbatim from upstream `transformers v4.56.0` (101 KB).
- Added [`data/__init__.py`](reference/transformers/src/transformers/data/__init__.py)
  as a minimal subset exporting only the data-collator symbols. Upstream's
  `__init__.py` also imports `.metrics` and `.processors`, which the Qwen path
  never uses; including them would pull in unused code/dependencies.
- Added a `.gitignore` negation for the vendored `data/` directory; the root
  `data` rule (meant for the project's dataset dir) was matching it.

`data_collator.py` imports only `..models.bert`, `..tokenization_utils_base`,
and `..utils`, all present in the vendored tree. Both files pass `py_compile`.

---

## Step 7 — First successful NPU training run

8 × Ascend NPU, synthetic data, 20 optimizer steps:

```bash
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
    --data_path logs/synth_qwen_data/synth_navsim.pkl \
    --data_root logs/synth_qwen_data/images \
    --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5 \
    --model_name_or_path /mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
    --device npu --ngpus 8 --batch_size 2 \
    --max_steps 20 --logging_steps 1 \
    --save_steps 100000 --save_total_limit 1
```

Environment / model:

- `torch 2.7.1+cpu`, `torch_npu 2.7.1.post4`, `transformers 4.56.0`
- 9,238.5M params total, 9,154.8M trainable (7B LLM + vision + SD1.5 UNet)
- script estimate: ~187.6 GiB optimizer + master weights (ZeRO-3 CPU offload)
- dataset: 4,096 synthetic samples; per-rank dataloader length 256

Trainer output:

| Metric | Value |
|---|---|
| `train_runtime` | 304.88 s (includes the final checkpoint save) |
| `train_samples_per_second` | **1.05** |
| `train_steps_per_second` | **0.066** |
| `train_loss` | 13.28 |
| steady-state step time | **~8.4 s/step** (progress bar at 20/20 before saving) |
| `ross_loss` | ~0.97–1.01, stable, no NaN |
| `action_loss` | 19.8 → 7.6 (random labels — not a learning signal) |

Notes:

- The step time starts at ~12 s (warm-up) and settles at ~8.3–8.4 s/step.
  The end-to-end 0.066 steps/s is dominated by checkpoint I/O, not compute.
- Despite `--save_steps 100000`, the Trainer writes `checkpoint-20` at the end,
  then the script writes the final model. The run dir reached **~137 GB**
  (~18 GB final model + ~120 GB checkpoint with ZeRO optimizer state).
  For throughput-only runs, delete the checkpoint or add a
  `--save_strategy no` option (not exposed by the launcher yet).
- `use_previous_actions=false` was accepted; the 2-segment assert passed.

---

## Step 8 — ROSS loss diagnosis

The 20-step run logged `ross_loss` ≈ 0.97–1.01 with no downward trend. A
diagnostic run and checkpoint inspection showed why that run was not a valid
learning test.

**First-run checkpoint inspection** (`checkpoint-20`, compared against the base
weights):

| Weight group | Result |
|---|---|
| LLM / ViT | changed (max ≈ 2.4e-4 / 1.2e-4) |
| SD1.5 UNet | bit-identical to the original bf16 weights |
| `denoiser.factor` | 6.6e29 (anomalous) |
| `denoiser.factor_a` | 4e-10 |

**Controlled 100-step diagnostic** (`lr=1e-4`, `warmup=10`, `save_strategy=no`,
same seed/data):

| Metric | Value |
|---|---|
| `train_runtime` | 806.98 s / 100 steps = **8.07 s/step** |
| `train_samples_per_second` | **1.983** |
| `train_loss` | 5.82 |
| `ross_loss` | 0.0006–0.032, typical ~0.003, no divergence |
| `action_loss` | 19.85 → ~4.8 |
| UNet `conv_in.weight` | 8,934 / 11,520 elements changed, max 1.47e-3 |
| UNet mid-attention | 384,613 / 1,638,400 elements changed, max 9.77e-4 |
| `denoiser.factor` / `factor_a` | -5.0e-4 / 3.6e-4 (sane) |

Conclusion:

- The ROSS branch **does train**; its loss can drop to ~1e-3 on the synthetic
  data. The 20-step run never gave it a usable learning rate (warmup 50 meant
  the LR peaked at 2e-5 on the last step), and the UNet updates were below bf16
  save resolution.
- The first run's `factor=6.6e29` was not reproduced. Treat that checkpoint as
  anomalous; if it recurs, restore the `ross_grad_clip` / NaN guards that
  existed in `eb82f31` (they were removed from HEAD).
- Use `lr` in the 5e-5–1e-4 range, `warmup_steps` 10–50, and several hundred
  steps for real training. For throughput-only runs, `--save_strategy no`
  avoids the 120 GB optimizer checkpoint (the final 18 GB model is still
  written).

---

## Current status

### Verified locally

- Launcher parses (`bash -n`), `--help` works, missing-argument validation works.
- Synthetic data: 256 scenes, correct pickle schema, image path resolution,
  504×280 → 180 tokens.
- The vendored Qwen model/config/processor files are identical to upstream
  `transformers v4.56.0`; `trainer.py` differs by one import style and the
  Qwen `__init__.py` adds the ROSS module reference.
- `diffusers 0.41.0` still exports every symbol the custom UNet imports
  (`FusedAttnProcessor2_0`, `USE_PEFT_BACKEND`, `FromOriginalModelMixin`,
  `get_mid_block`, `retrieve_timesteps`, `rescale_noise_cfg`).
- The four NPU runtime fixes from Step 5 pass `py_compile`.
- The added `transformers/data` files pass `py_compile`.

### Verified on the remote NPU host

- `torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`,
  `torch_npu==2.7.1.post4` installed; `torch.npu.is_available()` → True,
  8 devices.
- `transformers.data` + `transformers.trainer` import successfully after
  Step 6 (no reinstall needed).
- **20-step training on 8 NPU completed successfully** (Step 7): all four
  fixes exercised, `ross_loss` stable around 1.0, model saved.
- `diffusers 0.41.0` + the custom UNet, DeepSpeed ZeRO-3 CPU offload +
  `torch_npu`, and `torch.amp.autocast('npu')` all worked at runtime.
- The synthetic 504×280 images produced the expected 180-token grid.

---

## Remaining port work

Core NPU training works (Step 7). What remains:

| # | File:line | Problem | Planned fix |
|---|---|---|---|
| 1 | [`inference/qwen/inference_action_navsim_vava.py:100-104`](inference/qwen/inference_action_navsim_vava.py#L100-L104) | inference hardcodes `backend='nccl'` and `torch.cuda` | use `hccl` / `torch.npu` when NPU; only needed for evaluation |
| 2 | `use_previous_actions=true` | ROSS asserts 2 action segments | currently disabled; if history is required, generalize the assert/extraction to handle 4 segments |
| 3 | launcher | no way to disable checkpointing | add `--save_strategy`; a 20-step run otherwise writes ~137 GB including optimizer state |
| 4 | training recipe | 20 steps / warmup 50 is not a valid learning test | use lr 5e-5–1e-4, warmup 10–50, several hundred steps (Step 8) |

Resolved by Step 7 (no longer open): `diffusers 0.41.0` UNet compatibility,
DeepSpeed ZeRO-3 + `torch_npu`, device-aware autocast, and the 180-token grid.

---

## Appendix — files changed since `db45a6f`

| File | Status | Step |
|---|---|---|
| [`requirements_qwen.txt`](requirements_qwen.txt) | new | 1, 3, 4 |
| [`scripts/scripts_train/train_qwen_vla_navsim.sh`](scripts/scripts_train/train_qwen_vla_navsim.sh) | new | 1, 2 |
| [`tools/synth/synth_qwen_vla_data.py`](tools/synth/synth_qwen_vla_data.py) | new | 2 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py) | modified | 4 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py) | modified | 4 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py) | modified | 5 |
| [`reference/transformers/.../modeling_ross/denoiser_sd.py`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py) | modified | 5 |
| [`reference/transformers/.../modeling_qwen2_5_vl_ross.py`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py) | modified | 5 |
| [`utils/train_qwen_vla.py`](utils/train_qwen_vla.py) | modified | 5 |
| [`reference/transformers/src/transformers/data/__init__.py`](reference/transformers/src/transformers/data/__init__.py) | new | 6 |
| [`reference/transformers/src/transformers/data/data_collator.py`](reference/transformers/src/transformers/data/data_collator.py) | new (upstream v4.56.0) | 6 |
| [`.gitignore`](.gitignore) | modified | 6 |
| [`npu-port.md`](npu-port.md) | new | this document |

Generated but not committed: `logs/synth_qwen_data/` (gitignored).
