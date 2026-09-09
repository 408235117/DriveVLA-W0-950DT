#!/usr/bin/env bash
#
# VLA inference on NavSim test set (VAVA with previous actions).
#
# Usage:
#   bash scripts/scripts_infer/infer_navsim_vava.sh \
#       --emu_hub /data/models/DriveVLA-W0 \
#       --output_dir ./output_inference \
#       --train_meta_pkl ./data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl
#
# All paths have sensible defaults — override as needed.
#

set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# ============================================================
# Defaults
# ============================================================
DEFAULT_EMU_HUB="/data/models/DriveVLA-W0"
DEFAULT_OUTPUT_DIR="$ROOT/logs/infer_navsim_vava_$(date +%Y%m%d_%H%M%S)"
DEFAULT_TRAIN_META_PKL="$ROOT/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl"
DEFAULT_INPUT_NUM_FRAME="1"
DEFAULT_NGPUS=1
DEFAULT_MASTER_PORT=23458
DEFAULT_DEVICE="auto"
DEFAULT_MODEL_ROOT="/data/models"

# ============================================================
# Parse arguments
# ============================================================
EMU_HUB="$DEFAULT_EMU_HUB"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
TRAIN_META_PKL="$DEFAULT_TRAIN_META_PKL"
INPUT_NUM_FRAME="$DEFAULT_INPUT_NUM_FRAME"
MAX_SAMPLES=""
NGPUS="$DEFAULT_NGPUS"
MASTER_PORT="$DEFAULT_MASTER_PORT"
DEVICE="$DEFAULT_DEVICE"
MODEL_ROOT="$DEFAULT_MODEL_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emu_hub)           EMU_HUB="$2";           shift 2 ;;
    --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
    --train_meta_pkl)    TRAIN_META_PKL="$2";    shift 2 ;;
    --input_num_frame)   INPUT_NUM_FRAME="$2";   shift 2 ;;
    --max_samples)       MAX_SAMPLES="$2";        shift 2 ;;
    --ngpus)             NGPUS="$2";             shift 2 ;;
    --master_port)       MASTER_PORT="$2";       shift 2 ;;
    --device)            DEVICE="$2";            shift 2 ;;
    --model_root)        MODEL_ROOT="$2";        shift 2 ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options (all optional, defaults in parentheses):"
      echo "  --emu_hub           <path>  ($DEFAULT_EMU_HUB)"
      echo "  --output_dir        <path>  ($DEFAULT_OUTPUT_DIR)"
      echo "  --train_meta_pkl    <path>  ($DEFAULT_TRAIN_META_PKL)"
      echo "  --input_num_frame   <int>   (1)"
      echo "  --max_samples       <int>   — truncate to N samples per GPU (default: all)"
      echo "  --ngpus             <int>   (1)"
      echo "  --master_port       <int>   (23458)"
      echo "  --device            <str>   (auto) — auto, cuda, or npu"
      echo "  --model_root        <path>  ($DEFAULT_MODEL_ROOT) — base for vq/vision/fast_tokenizer"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

# ============================================================
# Device setup
# ============================================================
if [ "$DEVICE" = "npu" ]; then
  export INF_NAN_MODE_ENABLE=1
  export CLOSE_MATMUL_K_SHIFT=1
  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  export ACL_OP_DETERMINISTIC=1
  export ASCEND_LAUNCH_BLOCKING=1
  export TASK_QUEUE_ENABLE=0
  export FLAGS_npu_storage_format=0
elif [ "$DEVICE" = "cuda" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

export DEVICE
export PYTHONPATH="$ROOT:$ROOT/reference/Emu3:$PYTHONPATH"

# ============================================================
# Verify paths
# ============================================================
echo "=== Inference config ==="
echo "  emu_hub:           $EMU_HUB"
echo "  output_dir:        $OUTPUT_DIR"
echo "  train_meta_pkl:    $TRAIN_META_PKL"
echo "  input_num_frame:   $INPUT_NUM_FRAME"
[ -n "$MAX_SAMPLES" ] && echo "  max_samples:       $MAX_SAMPLES"
echo "  ngpus:             $NGPUS"
echo "  master_port:       $MASTER_PORT"
echo "  device:            $DEVICE"
echo "  model_root:        $MODEL_ROOT"
echo ""

for p in "$EMU_HUB" "$TRAIN_META_PKL"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: $p not found. Override via --${p##*/} or place the file at the default path."
    exit 1
  fi
done

# ============================================================
# Launch inference
# ============================================================
mkdir -p "$OUTPUT_DIR"

# Resolve helper paths from model_root
VQ_HUB="$MODEL_ROOT/Emu3-Stage1"
VISION_HUB="$MODEL_ROOT/Emu3-VisionTokenizer"
FAST_TOKENIZER="$MODEL_ROOT/physical-intelligence-fast"

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    inference/vla/inference_action_navsim_with_previous_action_last_VAVA.py \
    --emu_hub "$EMU_HUB" \
    --output_dir "$OUTPUT_DIR" \
    --train_meta_pkl "$TRAIN_META_PKL" \
    --input_num_frame "$INPUT_NUM_FRAME" \
    --vq_hub "$VQ_HUB" \
    --vision_hub "$VISION_HUB" \
    --fast_tokenizer "$FAST_TOKENIZER" \
    $([ -n "$MAX_SAMPLES" ] && echo "--max_samples $MAX_SAMPLES")

echo ""
echo "=== Inference done ==="
echo "Results at: $OUTPUT_DIR"

echo ""
echo "=== Evaluation ==="
python "$ROOT/scripts/scripts_infer/eval_action_metrics.py" "$OUTPUT_DIR"