# Setup: GPU / NPU

## 1. System Prerequisites

Create conda venv

```bash
conda create -n drivevla python=3.10
conda activate drivevla
```

<details open>
<summary><b>GPU</b></summary>

```bash
# CUDA 12.4+
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.5.7
```
</details>

<details>
<summary><b>NPU</b></summary>

```bash
# CANN 8.5.1 + torch_npu
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install torch_npu==2.7.1.post4
```
</details>

## 2. Shared Dependencies

```bash
pip install -r requirements.txt
pip install deepspeed scipy tensorboard wandb
```

`flash-attn` in `requirements.txt` is CUDA-only — NPU falls back to SDPA automatically.

## 3. Model & Data Download

```bash
# Set these to match your layout
MODEL_ROOT=/path/to/models
DATA_ROOT=/path/to/datasets
PROJECT_ROOT=/path/to/DriveVLA-W0

pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### Pretrained Models

```bash
mkdir -p "$MODEL_ROOT"

# Base VLM
huggingface-cli download --resume-download BAAI/Emu3-Stage1 \
    --local-dir "$MODEL_ROOT/Emu3-Stage1"

# Vision tokenizer
huggingface-cli download --resume-download BAAI/Emu3-VisionTokenizer \
    --local-dir "$MODEL_ROOT/Emu3-VisionTokenizer"

# Action tokenizer (FAST)
huggingface-cli download --resume-download physical-intelligence/fast \
    --local-dir "$MODEL_ROOT/physical-intelligence-fast"
```

### Training Data

Download from [Hugging Face](https://huggingface.co/liyingyan/DriveVLA-W0) into `$DATA_ROOT`:

- `train_vq_codes.zip` — VQ code indices (train)
- `test_vq_codes.zip` — VQ code indices (test)
- `navsim_emu_vla_256_144_trainval_pre_1s.pkl` — train/val metadata pickle
- `navsim_emu_vla_256_144_test_pre_1s.pkl` — test metadata pickle

## 4. End-to-End Pipeline (One Command)

For a fully automated run — data prep, training, and inference:

```bash
# GPU
bash scripts/setup_and_run_cuda.sh \
    --project_root /path/to/DriveVLA-W0 \
    --data_root /path/to/datasets \
    --model_root /path/to/models

# NPU
bash scripts/setup_and_run_npu.sh \
    --project_root /path/to/DriveVLA-W0 \
    --data_root /path/to/datasets \
    --model_root /path/to/models
```

### Custom flags

Need to override specific training flags? Call the training launcher directly:

```bash
MODEL_ROOT=/path/to/models
DATA_ROOT=/path/to/datasets

bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 \
    --batch_size 1 \
    --max_steps 200 \
    --save_steps 100 \
    --fp fp16 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --deterministic \
    --device npu
```

See `scripts/scripts_train/train_base_ar_withou_moe.sh --help` for all options.

## 5. Compare CUDA vs NPU

After running both pipelines, compare the training trajectories:

```bash
python scripts/plot_drivevla.py \
    --cuda_json /path/to/cuda/logs/.../trainer_state.json \
    --npu_json /path/to/npu/logs/.../trainer_state.json \
    -o ./comparison_plots
```

Generates `loss_comparison.png`, `relative_error.png`, and `relative_error_after50.png` in the output directory, plus a summary stats block.

## 6. Precision Alignment (NPU vs GPU)

For deterministic reproducibility across platforms, follow the step-by-step guide:

→ **[PRECISION_ALIGNMENT.md](PRECISION_ALIGNMENT.md)**
