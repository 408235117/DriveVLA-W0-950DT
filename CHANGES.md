# DriveVLA-W0: Key Changes by mingchang93

**87 commits** | **41 files changed** | **+3,934 / −124 lines** | 2026-07-18 → 2026-07-27

---

## 1. Ascend NPU Cross-Platform Support (Jul 18–19)

The foundational work: making the entire training + inference pipeline work on **both NVIDIA CUDA and Huawei Ascend NPU** from a single codebase.

| Commit | What | Why |
|--------|------|-----|
| `b2f29ff` | Device-agnostic training path with `utils/train_moe.py` wrappers | NPU uses `torch_npu`; CUDA uses standard `torch.cuda`. Wrappers (`device_synchronize`, `device_empty_cache`, `device_manual_seed_all`) abstract the difference so no platform branches exist in training logic |
| `1dc099f` | Enhanced device detection and setup | Auto-detect `torch_npu` availability; fall back gracefully when NPU drivers aren't present |
| `ba52cc0` | Improved NPU device detection and error handling | Early, clear error messages when NPU environment is misconfigured rather than cryptic crashes deep in training |
| `0440b43` | Suppress DeepSpeed env vars during `from_pretrained()` | DeepSpeed batch-size assertion fires before HuggingFace Trainer init on NPU — suppressing `DS_CONFIG`/`CONFIG_FILE` during model loading avoids the false positive |
| `e91021a` | Reproducibility settings for NPU: non-saturation mode, dropout control | NPU's deterministic ops differ from CUDA; enabling non-saturation mode and explicitly controlling dropout layers ensures bitwise-identical results across runs on the same platform |
| `7ac2c84` | Reproducibility helpers + cross-platform consistency | Seed initialization, deterministic algorithms, and stream synchronization so CUDA and NPU training trajectories are comparable |
| `65ca50c` | Device-aware attention implementation | FA2 is CUDA-only; auto-fallback to `"sdpa"` on NPU. Single `--attn_type` flag (`sdpa`/`fa2`/`eager`) with platform-aware defaults |
| `a0aafe1` | Refactor tensor dtypes to float32 for NPU compatibility | NPU's mixed-precision support is stricter than CUDA's; explicit float32 casts at trust boundaries prevent silent precision loss |
| `114717c` | Precision alignment guide link in SETUP.md | Documents the known precision gap between CUDA and NPU and how to measure it |
| `aa8a10a`, `9b64c01`, `79372e7`, `17cb8f0` | `requirements_ascend.txt`, torch_npu versioning | NPU dependencies are separate from CUDA (`flash-attn` is CUDA-only); version pinning to `torch_npu 2.7.1.post4` for CANN 8.5.1 compatibility |

**Design principle:** No `if cuda: ... elif npu: ...` branches in core logic. Platform differences are behind a thin wrapper layer.

---

## 2. Memory Optimization — 184K Vocabulary MoE Training (Jul 18)

The Qwen VLA model has a **184K-token vocabulary** with a Mixture-of-Experts language head. Materializing the full `[batch, seq_len, 184000]` logits tensor causes OOM even on 80GB GPUs.

| Commit | What | Why |
|--------|------|------|
| `2e66b1c` | Chunked cross-entropy for 184K vocab | Instead of computing all logits at once, process the LM head in chunks along the vocabulary dimension. Each chunk computes logits → softmax → cross-entropy for its slice, then discards the chunk. Peak memory drops from O(184K) to O(chunk_size) |
| `4422b59` | Chunked lm_head + loss: never materialize full logits | The key insight: accumulate loss across chunks without ever concatenating logits. Each chunk produces a scalar loss contribution |
| `5bb4c9f` | Fix dtype mismatch: cast chunk logits to float32 | NPU cross-entropy requires float32 input; mixed-precision logits cause silent incorrect results |
| `96ba4fc` | Use `hidden_states.dtype` instead of hardcoded float32 | Let the framework's autocast decide the compute dtype; only cast at the loss boundary |
| `2e27382` | Disable KV cache, set `expandable_segments` | KV cache is unnecessary during training (no autoregressive generation) and wastes memory; `expandable_segments` reduces CUDA fragmentation |

**Impact:** Training runs on the same hardware that previously OOM'd at batch size 1.

---

## 3. Training Infrastructure (Jul 18–25)

### CLI & Configurability

| Commit | What | Why |
|--------|------|------|
| `8270c99` | `--max_steps` CLI arg (default 4000) | Previously hardcoded; needed for quick experiments |
| `cfaf55b` | `--save_steps`, `--eval_strategy`, `--eval_steps` | Control checkpoint frequency and evaluation separately from logging |
| `75849bb` | `--logging_steps` CLI arg (default 10) | Decouple log frequency from save frequency |
| `e428c72` | `--zero_stage` CLI arg (2\|3, default 3) | ZeRO-3 for memory, ZeRO-2 for speed — tradeoff depends on model size |
| `4e593fd` | `--warmup_steps` CLI arg (default 50) | Linear warmup is critical for large MoE models; make it tunable |
| `8bf67f2` | `--fp` flag (bf16/fp16/fp32), disable TF32 on pre-Ampere | V100 doesn't support TF32; using it silently produces wrong results |

### LoggingTrainer & Data Handling

| Commit | What | Why |
|--------|------|------|
| `297ef46` | Pop `pre_action` and `cmd` before model forward | These are labels, not model inputs; passing them through causes shape mismatches |
| `db350a8` | Use `LoggingTrainer` so the pop takes effect | The default HuggingFace Trainer doesn't call the custom collate chain properly |
| `e4f76aa` | Handle non-model keys in trainer to prevent eval crashes | HuggingFace Trainer passes all dataset keys to `model.forward()`; non-tensor keys cause crashes |
| `32a7bc9` | Flush device stream in LoggingTrainer for accurate gradient reporting | NPU's async execution means gradient norms read before the stream completes show stale values; explicit sync fixes this |
| `5504f72` | Seed initialization for reproducibility | Deterministic weight initialization across runs |
| `29ece96` | LR to 1e-5, enhanced NPU reproducibility | Lower LR for stable MoE training; non-saturation mode for NPU determinism |
| `6c8b512` | Constant LR scheduler option | Some experiments need fixed LR for fair comparison |

### Data Verification

| Commit | What | Why |
|--------|------|------|
| `72685c1`, `c027816` | SHA256 data hash logging | Cross-platform verification: same data should produce identical hashes on CUDA and NPU. Catches data corruption or platform-specific serialization bugs |
| `034b92b` | Shuffle option for training data | Deterministic shuffle with seed for reproducibility; disable for controlled experiments |
| `17e9e51` | `--shuffle_train_data` arg in experiment scripts | Same shuffle flag exposed to shell launchers |

---

## 4. Data Pipeline — Pickle Path Fixing (Jul 18–20)

The NavSim dataset pickle files contained **hardcoded absolute paths** from the original training machine, making them unusable after relocation.

| Commit | What | Why |
|--------|------|------|
| `f3b7baf` | Script to fix hardcoded paths in pickle files | Recursively walks pickle structures, replaces old path prefixes with new ones |
| `bc592a6` | Handle nested lists (`image`, `pre_1s_image`) recursively | Image paths are stored in nested lists within scene dicts; flat replacement missed them |
| `1be9065` | Limit dry-run to 2 samples, handle list fields | Dry-run was printing entire dataset; 2 samples is enough to verify correctness |
| `6a61fee`, `3456cff` | Refactor path replacement logic | Cleaner separation: scan → replace → verify |
| `a169d91` | Add `--dry-run`, summary stats, fix tuple return bug | Tuple return from recursive walk was silently unpacking wrong; fixed with explicit return type |
| `0971057` | Symlink creation for legacy pickle paths | Some downstream code referenced the old paths directly; symlinks provide backward compatibility without duplicating data |
| `51005f6` | Remove stale fixed pickle files before saving | Overwriting existing files left partial writes on failure; remove first, then write atomically |
| `e81e4ef` | Refactor scene_dict_all to use scene names as keys | Dict-of-lists → dict-of-dicts; faster lookup by scene name, matches how downstream code actually accesses the data |
| `360ffe3` | Inline normalization statistics | NavSim normalization was in a separate file; inlining avoids path dependencies |

---

## 5. Inference Pipeline (Jul 26)

| Commit | What | Why |
|--------|------|------|
| `08010be` | Refactor logits processor init + action handling | VLA action tokens need special decoding (discrete action bins → continuous values); the logits processor constrains generation to valid action tokens only |
| `97b6adf` | Robust action decoding | Edge cases: incomplete sequences, tokens outside valid action range, multi-modal action distributions |
| `1627194` | Model path updates + data remapping | Paths changed between training and inference environments; remapping handles this transparently |
| `170e43b` | `--max_samples` arg for sample truncation | Quick validation runs on 10 samples instead of full dataset |
| `a658e15` | `--min_action_tokens` arg | Discard generations with fewer than N action tokens (incomplete outputs) |
| `96cbb41` | Evaluation script for VLA inference results | Compare predicted vs ground-truth trajectories; metrics: ADE, FDE, collision rate |
| `11c89b1` | Set GPU usage to 1 for inference | Multi-GPU inference with vLLM-like batched generation causes race conditions; single GPU is simpler and fast enough |
| `42d9168` | `--norm_config` arg | Normalization statistics differ between datasets; make it configurable |
| `e81d8a9` | Handle FA2 ImportError gracefully | Inference environment may not have `flash-attn` installed; fall back to `sdpa` |

---

## 6. Experiment Orchestration (Jul 24–27)

Shell scripts for reproducible, comparable CUDA vs NPU training runs.

| Commit | What | Why |
|--------|------|------|
| `5073419` | CUDA/NPU experiment scripts with bf16/fp16 configs | One script per platform × precision combination; shell `--device` arg dispatches to correct env |
| `75abb21`, `d2f59ac`, `6c8b512` | fp32 experiment scripts; refactor to unified fp16/fp32 launchers | Early experiments needed fp32 for debugging; later unified into single script with `--fp` flag |
| `a8735a9`, `4618446`, `7a829c5` | Step counts: 100 → 500 → 600 | Progressive refinement: 100 steps for smoke tests, 500 for meaningful convergence, 600 for alignment with paper baseline |
| `00e17fa` | Cleanup wait time between experiments | NPU requires explicit cache clearing between runs; a 30s wait prevents OOM from residual allocations |
| `12212f3`, `33de94a` | Timestamped output directories | Prevent accidental overwrite of previous experiment logs |
| `04f6e84` | End-to-end pipeline scripts (train → inference) | Single command runs training then inference on the checkpoint; eliminates manual steps |

---

## 7. Analysis & Visualization (Jul 27)

| Commit | What | Why |
|--------|------|------|
| `9dd7abc` | Inference output comparison with metrics and plotting | Compare CUDA vs NPU inference outputs: trajectory plots, action distribution histograms, per-dimension error |
| `3c1c0ee` | Additional metrics: cosine similarity, KL divergence, per-token accuracy | Beyond simple L2 distance — captures distributional differences between platform outputs |
| `a42269f` | CUDA vs NPU training trajectory comparison + plotting | Loss curves, gradient norms, learning rate schedule — overlaid for both platforms to identify divergence points |

---

## 8. Setup & Documentation (Jul 18–19)

| Commit | What | Why |
|--------|------|------|
| `4888e20`, `9b7bb3b` | README: split GPU & Ascend conda setup | Different conda channels, different PyTorch builds; clear separation avoids confusion |
| `b9567b4` | Core dependencies in README | `transformers`, `torch`, `deepspeed` minimum versions |
| `1006275` | SETUP.md with GPU + NPU environment instructions | Step-by-step: conda env, CANN toolkit, `torch_npu` install, verification commands |
| `81ddbef` | Consolidate `transformers[torch]` into `requirements.txt` | Single source of truth for pip deps |
| `faa74a8` | `.codegraph/` in `.gitignore` | Local code intelligence index; shouldn't be committed |
| `73cbbdc` | `CLAUDE.md` + `AGENTS.md` in `.gitignore` | Local-only project instructions for AI coding assistants |

---

## Timeline Summary

```
Jul 18 ──── Jul 19 ──── Jul 20 ──── Jul 24 ──── Jul 25 ──── Jul 26 ──── Jul 27
  │            │            │            │            │            │            │
  │ NPU        │ Reprodu-   │ Data       │ Experiment │ Precision  │ Inference  │ Analysis
  │ support    │ cibility   │ pipeline   │ scripts    │ + LR       │ pipeline   │ + plots
  │ OOM fix    │ Setup docs │ pickle fix │ bf16/fp16  │ tuning      │ action     │ CUDA vs
  │ CLI args   │            │ NavSim     │            │             │ decoding   │ NPU
  │            │            │            │            │             │            │ comparison
```

**Core arc:** Stand up NPU training → make it reproducible → fix the data pipeline → run controlled experiments → build inference → compare and analyze.