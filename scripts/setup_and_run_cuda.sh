#!/usr/bin/env bash
#
# End-to-end GPU pipeline: data → train → infer
#
# Prerequisites: run SETUP.md sections 1–3 first (torch, deps, model & data download).
#
# Usage:
#   bash scripts/setup_and_run_cuda.sh \
#       --project_root /path/to/DriveVLA-W0 \
#       --data_root /path/to/datasets \
#       --model_root /path/to/models
#
set -e

# ============================================================
# Parse required arguments
# ============================================================
PROJECT_ROOT=""
DATA_ROOT=""
MODEL_ROOT=""
BATCH_SIZE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_root) PROJECT_ROOT="$2"; shift 2 ;;
    --data_root)    DATA_ROOT="$2";    shift 2 ;;
    --model_root)   MODEL_ROOT="$2";   shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
    --help|-h)
      echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path> [--batch_size <int>]"
      echo ""
      echo "Required:"
      echo "  --project_root   Path to the DriveVLA-W0 repo"
      echo "  --data_root      Path to datasets (pickles + VQ code zips)"
      echo "  --model_root     Path to pretrained models (Emu3-Stage1, etc.)"
      echo ""
      echo "Optional:"
      echo "  --batch_size     Per-GPU train batch size (default 1)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path>"
      exit 1
      ;;
  esac
done

if [ -z "$PROJECT_ROOT" ] || [ -z "$DATA_ROOT" ] || [ -z "$MODEL_ROOT" ]; then
  echo "ERROR: --project_root, --data_root, and --model_root are required."
  echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path>"
  exit 1
fi

cd "$PROJECT_ROOT"

# ============================================================
# Derived paths
# ============================================================
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"
MODEL_PATH="$MODEL_ROOT/Emu3-Stage1"
TRAIN_PKL="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
TEST_PKL="$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl"
TRAIN_PKL_FIXED="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl"
TEST_PKL_FIXED="$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/logs/e2e_cuda_${TIMESTAMP}"

echo "============================================"
echo "DriveVLA-W0 — End-to-End GPU Pipeline"
echo "============================================"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  DATA_ROOT:    $DATA_ROOT"
echo "  MODEL_ROOT:   $MODEL_ROOT"
echo "  output:       $OUTPUT_DIR"
echo ""

# ============================================================
# Phase 1: Data Preparation
# ============================================================
echo "=== [1/3] Data Preparation ==="

apt-get update && apt-get install -y unzip

# Unpack VQ code zips
cd "$DATA_ROOT"
unzip -o train_vp_codes.zip
unzip -o test_vq_codes.zip

cd "$PROJECT_ROOT"

# Fix pickle paths for the local machine
python tools/fix_pickle_paths.py \
    "$TRAIN_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python tools/fix_pickle_paths.py \
    "$TEST_PKL" \
    --old_prefix /mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

# Move fixed pickles into place (idempotent)
[ -f "$TRAIN_PKL_FIXED" ] && mv "$TRAIN_PKL_FIXED" "$TRAIN_PKL"
[ -f "$TEST_PKL_FIXED" ] && mv "$TEST_PKL_FIXED" "$TEST_PKL"

echo ""

# ============================================================
# Phase 2: Training
# ============================================================
echo "=== [2/3] Training ==="

bash "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$TRAIN_PKL" \
    --test_data_path "$TEST_PKL" \
    --output_dir "$OUTPUT_DIR" \
    --ngpus 8 \
    --batch_size "$BATCH_SIZE" \
    --warmup_steps 100 \
    --logging_steps 1 \
    --device cuda \
    --attn_type fa2 \
    --log_data_hash \
    --deterministic \
    --shuffle_train_data false \
    --eval_strategy no \
    --eval_steps 10000 \
    --fp bf16 \
    --max_steps 600 \
    --save_steps 200 \
    --exp_name bf16_600steps

echo ""

# ============================================================
# Phase 3: Inference
# ============================================================
echo "=== [3/3] Inference ==="

# Resolve last checkpoint
CKPT_BASE="$OUTPUT_DIR/bf16_600steps"
LAST_CKPT=$(ls -d "$CKPT_BASE"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
EMU_HUB="${LAST_CKPT:-$CKPT_BASE}"

bash "$PROJECT_ROOT/scripts/scripts_infer/infer_navsim_vava.sh" \
    --emu_hub "$EMU_HUB" \
    --output_dir "$OUTPUT_DIR/json_output" \
    --train_meta_pkl "$TEST_PKL" \
    --input_num_frame 1 \
    --ngpus 1 \
    --device cuda

echo ""
echo "============================================"
echo "GPU pipeline complete."
echo "  Training:  $OUTPUT_DIR"
echo "  Inference: $OUTPUT_DIR/json_output"
echo "============================================"