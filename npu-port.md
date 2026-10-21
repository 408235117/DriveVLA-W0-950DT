# Qwen2.5-VL NPU Port — Work Log

**Base commit:** `db45a6f7f2d6e966b6b94166a33dcf8e1014804e`
**Branch:** `qwen-npu-port`
**Scope:** train the Qwen2.5-VL (ROSS) VLA backbone on Ascend NPU, starting with a
synthetic-data throughput run.

This document records every change made on the branch since the base commit, the
reason for it, and what is still outstanding.

---

## Accessing the NPU box

| Item | Value |
|---|---|
| SSH alias | `ott9-a57` |
| Host / user | `liteserver-hps-d235-00007.novalocal` / `root` |
| Repository | `/mnt/sfs_turbo/lmc/DriveVLA-W0` |
| tmux session | `lmc` (also `lmc-2`, `lmc-compile`) |
| Conda env | `drivevla` — active only inside the `lmc` session |
| Accelerators | 8 × Ascend NPU (`/dev/davinci0..7`) |
| Qwen weights | `/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct` |
| SD1.5 weights | `/mnt/sfs_turbo/models/stable-diffusion-v1-5` |
| Synthetic data | `logs/synth_qwen_data` (4,096 scenes), `logs/synth_qwen_data_large` (8,192) |

### Connect and attach

```bash
ssh ott9-a57
tmux attach -t lmc
```

### Run a command without attaching

```bash
ssh ott9-a57 "tmux send-keys -t lmc 'cd /mnt/sfs_turbo/lmc/DriveVLA-W0 && python -c \"import torch; print(torch.__version__)\"' C-m"
ssh ott9-a57 "tmux capture-pane -t lmc -p | tail -20"
```

For automation, `ssh -o BatchMode=yes ott9-a57 '<command>'` works (key auth is
configured) and returns the command's stdout.

### Launching runs — always use `tmux`

**Launch every training/eval run inside the `lmc` tmux session. Do not use
detached processes (`docker exec -d`, `nohup ... &`, background SSH jobs)
unless explicitly told otherwise.** The session runs inside the training
container with the env already active, and it is what the user monitors.

```bash
ssh ott9-a57 "tmux send-keys -t lmc 'cd /mnt/sfs_turbo/lmc/DriveVLA-W0 && bash run_100.sh 2>&1 | tee logs/run.log' C-m"
ssh ott9-a57 "tmux capture-pane -t lmc -p | tail -20"
```

To stop a job, kill its processes (`pkill -f train_qwen_vla.py`) — do not send
`C-c` to the shared pane.

### Change workflow

Every code change that a run will use follows exactly one path:

1. modify locally
2. commit locally **only when the user says so** — never commit on your own
3. push **from this machine**
4. pull on the box
5. run, and add the run directory + base commit to the Run index

**The box is pull-only.** Never `git push` from it. **Never copy files to the
box** (`scp`, `rsync`, `tar | ssh`) either — every change must arrive through
commit → push → pull so there is a record of what actually ran. If this
machine cannot reach GitHub, stop and ask rather than routing around the
push.

### Environment caveat

A plain non-interactive SSH shell is **not** the training environment:

```text
ssh ott9-a57 'which python'   -> /usr/bin/python   (3.11.6, no torch)
```

The `drivevla` env is active only inside the `lmc` tmux session
(`which python` → `/usr/local/python3.12.13/bin/python`,
`CONDA_PREFIX=/root/miniconda3/envs/drivevla`). Run Python/torch work through
tmux, or use the exact interpreter path from inside that session.

`/mnt/sfs_turbo` is shared with the SSH shell, so `git`, `ls`, `ps`, `du`, and
file inspection can be done directly over SSH.

### Verify the box

```bash
ssh ott9-a57 'ls /dev/davinci* | head; nproc'
ssh ott9-a57 "tmux send-keys -t lmc 'python -c \"import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())\"' C-m"
```

Expected: eight devices and `(2.7.1+cpu, True, 8)`.

**Do not send `C-c`** to the `lmc` pane unless you intend to kill the running
job — the session is shared.

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
| Training requires **Stable Diffusion 1.5** | UNet at `denoiser_sd.py:40`, VAE at `modeling_qwen2_5_vl_ross.py:71-81`, scheduler at `denoiser_sd.py:62`; `--sd_model_path` must end in `/unet` because paths are derived with `.replace("/unet", ...)`. |
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
sharded. The generator default is 64 scenes (`--num_scenes`); regenerate with
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

## Step 5 — NPU runtime fixes (committed as `2e40d2c`)

Triggered by the first real launch on the NPU host:

```
ModuleNotFoundError: No module named 'flash_attn'
  at qwenvl/train/trainer.py:7
```

Five fixes were applied so the training path can start on NPU and reach a
forward/backward pass:

| # | File | Change |
|---|---|---|
| 1 | [`qwenvl/train/trainer.py:7`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py#L7) | `flash_attn` import wrapped in `try/except ImportError` (`flash_attn_varlen_func = None`); `_flash_attention_forward` now raises a clear error only if `fa2` is actually used. `sdpa` runs without flash-attn. |
| 2 | [`modeling_ross/denoiser_sd.py:25-31`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L25-L31) | `negative_prompt_path` default changed from the author's `/mnt/vdb1/...` path to `None`, resolved relative to `__file__` (`negative_prompt_sd15.pt` next to the module). Added `import os`. |
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
- **Caveat (added in Step 10):** the synthetic `pre_1s_*` streams were
  byte-identical to the current-frame streams, so this objective was a
  near-identity reconstruction rather than `t → t+1`. The ~1e-3 loss does not
  demonstrate the intended ROSS task.
- The first run's `factor=6.6e29` was not reproduced. Treat that checkpoint as
  anomalous; if it recurs, restore the `ross_grad_clip` / NaN guards that
  existed in `eb82f31` (they were removed from HEAD).
- Use `lr` in the 5e-5–1e-4 range, `warmup_steps` 10–50, and several hundred
  steps for real training. For throughput-only runs, `--save_strategy no`
  avoids the 120 GB optimizer checkpoint (the final 18 GB model is still
  written).

---

## Step 9 — Larger-dataset confirmation (interrupted at 70/300)

Regenerated 8,192 scenes / 98,304 images (455 MB) at
`logs/synth_qwen_data_large`, then ran 300 steps with `lr=1e-4`,
`warmup=10`, `save_strategy=no`.

The run was **interrupted by SIGINT at step 70/300** (9:41 elapsed). Observed
before the interrupt:

| Metric | Value |
|---|---|
| `action_loss` | 20.39 → 5.21 |
| `ross_loss` | 0.00055–0.0129, typical ~0.002 |
| `grad_norm` | ~9.7 |
| LR at step 70 | 9.02e-5 |
| step time | 7.72 s/step |

This confirms the 100-step diagnostic on a 2× larger dataset: the ROSS loss is
stable, low (~2e-3), and three orders of magnitude below the first run's ~1.0.
The run directory is empty because the process was killed before the final
model save.

**Caveat (Step 10):** this run used the same degenerate pre/cur pairing as
Step 8 — `pre_1s_image == image` in every scene. The low loss therefore
reflects a near-identity reconstruction, not `t → t+1` prediction.

To complete the full 300 steps, rerun `run_confirm.sh` on the host (or the
equivalent command); the SIGINT came from the terminal, not the training code.

---

## Step 10 — Fix synthetic `pre_1s_*` and re-run (60 steps)

Auditing the synthetic data against the upstream generator found that every
scene had `pre_1s_image == image` and `pre_1s_action == action`:

- [`synth_qwen_vla_data.py`](tools/synth/synth_qwen_vla_data.py) copied the current-frame lists into the `pre_1s_*` fields (now fixed at [:85-87](tools/synth/synth_qwen_vla_data.py#L85-L87), self-check at [:91-98](tools/synth/synth_qwen_vla_data.py#L91-L98)).
- Upstream does not: [`pickle_generation_navsim_pre_1s.py:143-145`](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L143-L145) anchors `pre_1s_*` on `scene[i-2]`, and Phase 2 of that script explicitly repairs identical pairs ([:236](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L236), [:260-277](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L260-L277)).

Because the ROSS branch conditions on the pre segment and reconstructs the
current frame ([`modeling_qwen2_5_vl_ross.py:339-342`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L339-L342)),
the copy reduced the objective to "reconstruct the frame you conditioned on."

Fix (`e988fef`): `pre_1s_*` is now the same window anchored `PRE_SHIFT = 2`
frames earlier, with an assertion in the generator that fails if the pair
collapses back to a copy.

Verification (inside the training container, against the real dataset class):

| Check | Result |
|---|---|
| `pre_1s_image == image` | 0 / 4096 and 0 / 8192 after regeneration |
| shift alignment `pre[k] == image[k-2]` | all scenes |
| schema / image paths / 180-token grid | unchanged, 0 violations |
| ROSS segment asserts | image T=2, action T=2 |

Regenerated `logs/synth_qwen_data` (4,096 scenes) and
`logs/synth_qwen_data_large` (8,192) on the NPU host.

60-step run on the corrected data (`logs/ross_short_20261020`, lr 1e-4,
warmup 10, `save_strategy no`):

| Metric | Value |
|---|---|
| `train_runtime` | 523.3 s = **8.72 s/step** |
| `train_samples_per_second` | 1.835 |
| `ross_loss` | 0.972 → ~1.0, **flat across all 60 steps** |
| `action_loss` | 19.67 → ~5.0 |
| `grad_norm` | 67 → ~9 |
| final model | 18 GB |

The corrected objective sits at the epsilon-prediction baseline (~1.0, i.e.
predicting zero noise) and does not move within 60 steps. This confirms the
pipeline on non-degenerate inputs but is **not** a learning result; the earlier
~1e-3 losses were an artifact of the identity task. A few hundred steps are
needed to tell whether the ROSS branch fits the real objective.

**Corrected by Step 11:** the flat ~1.0 was not an insufficient-steps effect.
`factor` was frozen at a huge garbage value, which swamped the UNet input and
pinned the loss at the unconditional baseline. With the gate fixed, the first
step already reports `ross_loss` ≈ 0.0019 (Step 11). The ~1e-3 losses on the
degenerate data remain an artifact of the identity task, but the ~1.0 measured
here was the gate bug.

---

## Step 11 — ROSS conditioning gates were initialized from garbage

The two gates that scale the VLM conditioning into the SD1.5 UNet
([`denoiser_sd.py:53,60`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L53))
are bare `nn.Parameter`s created with `torch.tensor(0.)`. Values observed at
load:

| Run | `factor` | `factor_a` |
|---|---|---|
| 20-step (Step 7), saved | 6.6e29 | — |
| 60-step corrected, saved | −4.97e−13 | −1.78e24 |
| 100-step corrected, saved | −1.95e34 | 0.0069 |
| 3-step diagnostic | 6.36e−33 | 6.36e−33 |
| 3-step verification (after fix) | 3.9e−37 | **2.62e38** |

The `[GATE]` prints in `setup_vla_model_and_tokenizer` show the value is
already wrong immediately after `from_pretrained`, before DeepSpeed or the
Trainer exist.

Root cause:

- `from_pretrained` builds the model under `init_empty_weights()`
  ([`modeling_utils.py:4494`](reference/transformers/src/transformers/modeling_utils.py#L4494)),
  so the `torch.tensor(0.)` initializer is discarded and the parameter is
  created on the meta device.
- Neither gate exists in the Qwen checkpoint, so both are missing keys.
- `PreTrainedModel._init_weights` does not handle bare `nn.Parameter`s; its
  docstring says a model adding one must override it
  ([`modeling_utils.py:3028-3032`](reference/transformers/src/transformers/modeling_utils.py#L3028-L3032)).
- The missing keys are therefore materialized from uninitialized memory.

Consequences depended on the garbage value:

- Large (1e16–1e38): `factor * z` swamps the UNet input; the first GroupNorm
  normalizes the scale away, `ross_loss` sits at ~1.0, and the gradient with
  respect to the gate vanishes. The gate stays at that value for the whole run
  (confirmed: identical across all 100 steps of the Step 10 rerun).
- Tiny (≈1e−33): conditioning is effectively off, and the gates *do* move.

The conditioning tensors themselves were healthy throughout:
`z_absmean ≈ 0.18`, `za_absmean ≈ 0.23`.

Fix (`c024db1`): zero both gates explicitly after `from_pretrained`, restoring
the intended zero-init.

Verification — 3 steps after the fix (`logs/ross_gate_fixed_20261020`):

| Stage / step | `factor` | `factor_a` | `cond_absmean` | `ross_loss` |
|---|---|---|---|---|
| after `from_pretrained` | 3.9e−37 | 2.6e+38 | — | — |
| after zero-init | 0 | 0 | — | — |
| step 1 | 0 | 0 | 0 | 0.0019 |
| step 3 | −2.43e−06 | −6.89e−07 | 4.7e−07 | 0.0064 |

Both gates now start at zero and begin moving, so the gating is trainable.

The temporary `[GATE]` stage prints and the per-step `z_absmean` /
`za_absmean` / `cond_absmean` logging are still in the tree; remove them once
the fix is confirmed over a longer run.

---

## Step 12 — Throughput baseline (batch-size sweep)

Measured on 8 × Ascend 950DT with the current harness (ZeRO-3 CPU offload,
sdpa, bf16, gradient checkpointing, `model_max_length=4096`, 504×280 images,
synthetic 4,096-scene set). Code: `fbae548` (per-step timing instrumentation).

Sizes 2–16 ran 50 steps; 32–68 ran 20 steps (ceiling scan). Metrics are
per-step medians from the steady-state portion (first 5 steps dropped);
`step_time_s` is measured around a device sync.

| per-device bs | global batch | step median | samples/s median | samples/s max | peak HBM/device |
|---|---|---|---|---|---|
| 2 | 16 | 8.20 s | 1.96 | 2.02 | — |
| 4 | 32 | 8.76 s | 3.67 | 3.96 | — |
| 8 | 64 | 9.65 s | 6.64 | 7.20 | — |
| 16 | 128 | 11.39 s | 11.29 | 11.87 | — |
| 32 | 256 | 15.93 s | 16.14 | 16.60 | 96,550 MB (98.2%) |
| 64 | 512 | 27.96 s | 18.42 | 19.87 | 98,214 MB (99.9%) |
| 66 | 528 | 26.95 s | 19.75 | 20.66 | 93,471 MB (95.1%) |
| **68** | **544** | **27.11 s** | **20.49** | **21.32** | **98,285 MB (100.0%)** |
| 70 | 560 | OOM after 5 steps | — | — | 98,293 MB |
| 128 | 1024 | OOM at allocation (30.7 GiB) | — | — | — |

**Baseline:** the current harness peaks at **≈20.5 samples/s** (bs=68, global
batch 544). Recommended operating point is **bs=66** (19.75 samples/s, ~5 %
HBM headroom) or **bs=64** (18.42 samples/s) for longer runs.

Notes:

- `bs=1` is not testable — the ROSS forward squeezes away the batch dim.
- Throughput flattens above bs=32: doubling the batch buys ~14 % (32→64) and
  ~11 % (64→68). The step is dominated by fixed per-step cost (ZeRO-3 CPU
  optimizer, VAE encode, SD1.5 UNet); activation memory limits the batch, not
  parameter count.
- `bs=70` completed 5 steps before OOM, so fitting at step 1 does not prove a
  size is stable. HBM figures are 20 s maxima and under-report transients
  (bs=64 read 99.9 % while bs=66 read 95.1 %), so treat everything above
  bs=64 as sitting at the wall.
- Each run writes an 18 GB final model; the harness has no skip-save flag.

Run directories: `logs/bs_sweep50_20261020/`, `logs/bs_ceiling_20261020/`,
`logs/bs_fine_20261020/`. `logs/bs_sweep_20261020/` is a killed 12-step
attempt and should be ignored.

---

## Run index

Every run records the commit it is based on. "Code" is the commit whose tree
the run actually executed; where a directly copied file was used, both the
file's commit and the box `HEAD` are noted.

| Run directory | Step | Code | Steps |
|---|---|---|---|
| `train_qwen_vla_20261019_091442` | 7 | `67292e7` | 20 |
| `diag_ross_lr1e-4` | 8 | `67292e7` | 100 |
| `ross_confirm_300` | 9 | `67292e7` | 70 / 300 (interrupted) |
| `ross_short_20261020` | 10 | `e988fef` | 60 |
| `ross_100_20261020` | 10 | `e988fef` | 100 |
| `ross_100_gatelog_20261020` | 11 | `89edc4e` file on `c409ca4` HEAD (direct copy) | 100 (killed) |
| `ross_gatelog2_20261020` | 11 | `89edc4e` file on `c409ca4` HEAD (direct copy) | 100 |
| `ross_gate_diag_20261020` | 11 | `016a83f` | 3 |
| `ross_gate_fixed_20261020` | 11 | `c024db1` file on `016a83f` HEAD (direct copy) | 3 |
| `bs_sweep50_20261020/bs{2,4,8,16}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 50 |
| `bs_ceiling_20261020/bs{32,64,128}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 20 (128 OOM) |
| `bs_fine_20261020/bs{66,68,70}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 20 (70 OOM at step 5) |

`train_qwen_vla_20261019_085720` and `_090116` are earlier failed attempts
with no `trainer_state.json`.

---

## Current status

### Verified locally

- Launcher parses (`bash -n`), `--help` works, missing-argument validation works.
- Synthetic data: 256 scenes, correct pickle schema, image path resolution,
  504×280 → 180 tokens.
- `pre_1s_*` is a 2-frame-shifted view of the window, not a copy (Step 10);
  0 / 4096 and 0 / 8192 duplicate pairs after regeneration on the NPU host.
- The vendored Qwen model/config/processor files are identical to upstream
  `transformers v4.56.0`; `trainer.py` differs by one import style and the
  Qwen `__init__.py` adds the ROSS module reference.
- `diffusers 0.41.0` still exports every symbol the custom UNet imports
  (`FusedAttnProcessor2_0`, `USE_PEFT_BACKEND`, `FromOriginalModelMixin`,
  `get_mid_block`, `retrieve_timesteps`, `rescale_noise_cfg`).
- The five NPU runtime fixes from Step 5 pass `py_compile`.
- The added `transformers/data` files pass `py_compile`.

### Verified on the remote NPU host

- `torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`,
  `torch_npu==2.7.1.post4` installed; `torch.npu.is_available()` → True,
  8 devices.
- `transformers.data` + `transformers.trainer` import successfully after
  Step 6 (no reinstall needed).
- **20-step training on 8 NPU completed successfully** (Step 7): all five
  fixes exercised, `ross_loss` stable around 1.0, model saved.
- **60-step training on 8 NPU with corrected synthetic data** completed
  (Step 10): 8.72 s/step, ROSS loss flat at ~1.0, model saved.
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
| 3 | launcher | no way to disable checkpointing | add `--save_strategy`; a 20-step run otherwise writes ~137 GB including optimizer state. The ad-hoc `run_confirm.sh` / `run_short.sh` already pass `--save_strategy no`, leaving only the 18 GB final model |
| 4 | training recipe | 20 steps / warmup 50 is not a valid learning test | use lr 5e-5–1e-4, warmup 10–50, several hundred steps (Steps 8, 10); on corrected data the ROSS loss is still at baseline after 60 steps (Step 10) |

Resolved by Step 7 (no longer open): `diffusers 0.41.0` UNet compatibility,
DeepSpeed ZeRO-3 + `torch_npu`, device-aware autocast, and the 180-token grid.

---

## Appendix — files changed since `db45a6f`

| File | Status | Step |
|---|---|---|
| [`requirements_qwen.txt`](requirements_qwen.txt) | new | 1, 3, 4 |
| [`scripts/scripts_train/train_qwen_vla_navsim.sh`](scripts/scripts_train/train_qwen_vla_navsim.sh) | new | 1, 2 |
| [`tools/synth/synth_qwen_vla_data.py`](tools/synth/synth_qwen_vla_data.py) | new | 2, 10 |
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
