# Setup: GPU / NPU Precision Alignment

This guide covers environment setup, data preparation, and launching training on **NVIDIA GPU (CUDA)** and **Ascend NPU (CANN)** with deterministic precision alignment.

---

## 1. System Prerequisites

<!-- ============= GPU ============= -->
<details open>
<summary><b>GPU</b> — CUDA 12.4+</summary>

```bash
# Conda environment
conda create -n drivevla python=3.10
conda activate drivevla

# PyTorch (CUDA 12.4)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Flash-Attention (CUDA only)
pip install flash-attn==2.5.7

# Shared deps
pip install -r requirements.txt
pip install deepspeed scipy tensorboard wandb
```
</details>

<!-- ============= NPU ============= -->
<details>
<summary><b>NPU</b> — Ascend CANN 8.5.1</summary>

```bash
# Conda environment
conda create -n drivevla python=3.10
conda activate drivevla

# PyTorch + torch_npu
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install torch_npu==2.7.1.post4

# NPU deps (no flash-attn — falls back to SDPA automatically)
pip install -r requirements_ascend.txt
pip install deepspeed scipy tensorboard wandb
```
</details>

> `flash-attn` in `requirements.txt` is CUDA-only. On NPU the model loader falls back to `"sdpa"` — no action needed.

---

## 2. Model & Data Preparation

All models and data are downloaded from [Hugging Face — liyingyan/DriveVLA-W0](https://huggingface.co/liyingyan/DriveVLA-W0/tree/main).

```bash
# Install tools
apt-get update && apt-get install -y unzip git-lfs
pip install -U modelscope huggingface_hub

# Use HF mirror for faster downloads (domestic mainland China)
export HF_ENDPOINT=https://hf-mirror.com

# ──────────────────────────────────
# Set these to your actual paths
MODEL_ROOT=/data/models                  # where all models live
DATA_ROOT=/data/models/DriveVLA-W0       # where data (zips & pickles) live
REPO_ROOT=/data/lmc/DriveVLA-W0          # where the repo is cloned
# ──────────────────────────────────

mkdir -p "$MODEL_ROOT" "$DATA_ROOT"

# ─── 2.1 Pretrained Models ───

# Emu3-Stage1 — pretrained base, used as --model_name_or_path
# https://huggingface.co/BAAI/Emu3-Stage1
modelscope download --model BAAI/Emu3-Stage1 --local_dir "$MODEL_ROOT/Emu3-Stage1"

# Emu3-VisionTokenizer — vision tokenizer for encoding/decoding image tokens
# https://huggingface.co/BAAI/Emu3-VisionTokenizer
modelscope download --model BAAI/Emu3-VisionTokenizer --local_dir "$MODEL_ROOT/Emu3-VisionTokenizer"

# Physical Intelligence FAST — action tokenizer / policy head backbone
# https://huggingface.co/physical-intelligence/fast
hf download physical-intelligence/fast --local-dir "$MODEL_ROOT/physical-intelligence-fast"

# Emu3_NuPlan_Pretrain_Cktps — NuPlan-pretrained checkpoint (Stage 2 starting point)
# https://huggingface.co/liyingyan/DriveVLA-W0/tree/main/Emu3_NuPlan_Pretrain_Cktps
hf download liyingyan/DriveVLA-W0 Emu3_NuPlan_Pretrain_Cktps \
    --local-dir "$MODEL_ROOT/Emu3_NuPlan_Pretrain_Cktps"
```

> **Model dependency by pipeline stage:**
> - **Stage 1** (`train_base_ar_withou_moe.sh`): needs only `Emu3-Stage1`.
> - **Stage 2** (`train_navsim_ar.sh` / `train_navsim_query_based.sh` / `train_navsim_flow_matching.sh`): starts from `Emu3_NuPlan_Pretrain_Cktps` (the Stage 1 output trained on NuPlan), then finetunes on NavSim data.
> - **Inference**: adds `Emu3-VisionTokenizer` for image token decode. The fine-tuned DriveVLA-W0 checkpoints (`Emu3_Flow_Matching_Action_Expert_PDMS_87.2`, etc.) are pre-trained inference-only variants — skip unless evaluating.

```bash
# ─── 2.2 Data from DriveVLA-W0 HF Repo ───
# Pickle files (training/validation/test metadata)
hf download liyingyan/DriveVLA-W0 navsim_emu_vla_256_144_trainval_pre_1s.pkl \
    --local-dir "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 navsim_emu_vla_256_144_test_pre_1s.pkl \
    --local-dir "$DATA_ROOT"

# VQ code zips
hf download liyingyan/DriveVLA-W0 train_vp_codes.zip \
    --local-dir "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 test_vq_codes.zip \
    --local-dir "$DATA_ROOT"

# Unpack VQ code zips
unzip train_vp_codes.zip     # → data/navsim/processed_data/trainval_vq_codes_256_144/
unzip test_vq_codes.zip      # → data/navsim/processed_data/test_vq_codes_256_144/

cd "$REPO_ROOT"

# ─── 2.3 Fix Pickle Paths ───
# Each pickle stores absolute data-root prefixes from the original machine.
# Rewrite them so the loader finds files under DATA_ROOT.
python tools/fix_pickle_paths.py \
    "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python tools/fix_pickle_paths.py \
    "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl" \
    --old_prefix /mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"
```

After fixing, you'll have:
```
/data/models/
├── Emu3-Stage1/                             # BAAI base model (Stage 1)
├── Emu3-NuPlan-Pretrain-Cktps/              # after NuPlan pretrain (Stage 2 start)
├── Emu3-VisionTokenizer/                    # BAAI vision tokenizer (inference only)
├── physical-intelligence-fast/              # PI action tokenizer
└── DriveVLA-W0/
    ├── navsim_emu_vla_256_144_trainval_pre_1s.pkl        # original (from HF)
    ├── navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl  # after fix_pickle_paths
    ├── navsim_emu_vla_256_144_test_pre_1s.pkl             # original (from HF)
    ├── navsim_emu_vla_256_144_test_pre_1s_fixed.pkl       # after fix_pickle_paths
    ├── train_vp_codes.zip                                 # original zip (optional to keep)
    ├── test_vq_codes.zip                                  # original zip (optional to keep)
    └── data/navsim/processed_data/
        ├── trainval_vq_codes_256_144/        # from train_vp_codes.zip
        └── test_vq_codes_256_144/            # from test_vq_codes.zip
```

---

## 3. Stage 1: Base AR Training (from Emu3-Stage1)

Trains the world model on NavSim data starting from `Emu3-Stage1`. This is the precision-alignment entry point.

### 3.1 GPU (CUDA)

```bash
cd "$REPO_ROOT"

bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 \
    --batch_size 1 \
    --max_steps 200 \
    --save_steps 100 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --device cuda \
    --fp bf16 \
    --log_data_hash \
    --deterministic \
    --eval_strategy no \
    --eval_steps 10000 \
    --skip_inference
```

### 3.2 NPU (Ascend)

```bash
cd "$REPO_ROOT"

bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 \
    --batch_size 1 \
    --max_steps 200 \
    --save_steps 100 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --device npu \
    --fp bf16 \
    --log_data_hash \
    --deterministic \
    --eval_strategy no \
    --eval_steps 10000 \
    --skip_inference
```

All other flags (`--fp bf16`, `--deterministic`, `--log_data_hash`, `--warmup_steps 0`, `--logging_steps 1`) are identical across platforms.

Both flags `--log_data_hash` and `--deterministic` are **on for alignment** — they ensure identical data ordering and reproducible computation. Remove `--deterministic` for production throughput.

### 3.3 Verifying Alignment

1. Compare the loss trajectory — stored at `$REPO_ROOT/logs/train_base_ar/train_base_ar/trainer_state.json`. Compare step-by-step loss values between GPU and NPU runs.
2. For deeper debugging, see **[PRECISION_ALIGNMENT.md](PRECISION_ALIGNMENT.md)**.

---

## 4. Quick Reference

```bash
# ─── Variable cheat-sheet ───
MODEL_ROOT=/data/models
DATA_ROOT=/data/models/DriveVLA-W0
REPO_ROOT=/data/lmc/DriveVLA-W0

# ─── Download models ───
pip install -U modelscope huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
modelscope download --model BAAI/Emu3-Stage1 --local_dir "$MODEL_ROOT/Emu3-Stage1"
modelscope download --model BAAI/Emu3-VisionTokenizer --local_dir "$MODEL_ROOT/Emu3-VisionTokenizer"
hf download physical-intelligence/fast --local-dir "$MODEL_ROOT/physical-intelligence-fast"
hf download liyingyan/DriveVLA-W0 Emu3_NuPlan_Pretrain_Cktps --local-dir "$MODEL_ROOT/Emu3_NuPlan_Pretrain_Cktps"

# ─── Download data (one-time) ───
cd "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 navsim_emu_vla_256_144_trainval_pre_1s.pkl --local-dir "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 navsim_emu_vla_256_144_test_pre_1s.pkl --local-dir "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 train_vp_codes.zip --local-dir "$DATA_ROOT"
hf download liyingyan/DriveVLA-W0 test_vq_codes.zip --local-dir "$DATA_ROOT"
unzip train_vp_codes.zip && unzip test_vq_codes.zip
cd "$REPO_ROOT"
python tools/fix_pickle_paths.py "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl" --new_prefix "$DATA_ROOT/data/navsim/processed_data"
python tools/fix_pickle_paths.py "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl" --old_prefix /mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data --new_prefix "$DATA_ROOT/data/navsim/processed_data"

# ─── Stage 1: GPU training (from Emu3-Stage1) ───
bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 --batch_size 1 --max_steps 200 --save_steps 100 \
    --warmup_steps 0 --logging_steps 1 \
    --device cuda --fp bf16 --deterministic --eval_strategy no --skip_inference

# ─── Stage 1: NPU training (from Emu3-Stage1) ───
bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 --batch_size 1 --max_steps 200 --save_steps 100 \
    --warmup_steps 0 --logging_steps 1 \
    --device npu --fp bf16 --deterministic --eval_strategy no --skip_inference
```

See `scripts/scripts_train/train_base_ar_withou_moe.sh --help` for all available options.
