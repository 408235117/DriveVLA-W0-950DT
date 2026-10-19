#!/usr/bin/env bash
#
# Qwen2.5-VL (ROSS) VLA training for DriveVLA-W0.
#
# Usage:
#   bash scripts/scripts_train/train_qwen_vla_navsim.sh \
#       --data_path /path/to/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
#       --data_root /path/to/sensor_blobs/trainval \
#       --sd_model_path /path/to/stable-diffusion-v1-5 \
#       --device npu
#
#   CUDA:   same command with --device cuda (or auto)
#   Resume: rerun with the same --output_dir/--exp_name; train_qwen_vla.py
#           automatically resumes when checkpoint-* exists in the run dir.
#
# Before running:
#   pip install -r requirements_qwen.txt
#   (torch / torch_npu must already be installed — see SETUP.md / requirements_ascend.txt)
#
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Defaults
# ============================================================
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
SD_MODEL_PATH=""
ACTION_TOKENIZER_PATH="$ROOT/configs/fast"
DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json"
DEEPSPEED_CONFIG_EXPLICIT=false

DATA_PATH=""
DATA_ROOT=""
DATASET_TYPE="navsim2va_ross"
OUTPUT_DIR="$ROOT/logs/train_qwen_vla_$TIMESTAMP"
EXP_NAME="navsim"

NGPUS=8
MASTER_PORT=23458
BATCH_SIZE=2
GRAD_ACCUM=1
NUM_WORKERS=12
FP="bf16"
ATTN_TYPE="sdpa"
DEVICE="auto"

MAX_STEPS=4000
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10
WARMUP_STEPS=50
LR="5e-5"
WEIGHT_DECAY="0.1"
MAX_GRAD_NORM="5.0"
MODEL_MAX_LENGTH=4096
ZERO_STAGE=3
EVAL_STRATEGY="no"
REPORT_TO="tensorboard"

# ROSS forward hardcodes a 10x18 = 180-token image grid
# (reference/transformers/.../modeling_qwen2_5_vl_ross.py:359 + n_patches=180).
# For 16:9 camera images (1600x900 / 1920x1080) max_pixels=28*28*192 yields
# exactly 180 tokens. The DataArguments default 28*28*576 yields 576 and will
# fail the reshape. Do not point data_root at 256x144 previews (45 tokens).
MAX_PIXELS=150528
MIN_PIXELS=12544

FUTURE_NUMS=8
CUR_FRAME_IDX=3
ACTION_DIM=3
# The ROSS forward asserts exactly 2 action segments. With cur_frame_idx > 0,
# enabling history adds two more (pre-user + cur-user) and trips the assert
# (modeling_qwen2_5_vl_ross.py:335-336). Keep false unless the model is patched.
USE_PREVIOUS_ACTIONS=false
TUNE_MM_LLM=true
TUNE_MM_MLP=true
TUNE_MM_VISION=true

SEED=42
SHUFFLE=true
DETERMINISTIC=false
LOG_DATA_HASH=false

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --data_path <pkl>            NavSim train pickle (navsim_emu_vla_256_144_trainval_pre_1s.pkl)
  --data_root <dir>            Camera root containing <log>/CAM_F0/<frame>.jpg
  --sd_model_path <dir>        Stable Diffusion 1.5 dir (parent with unet/, vae/,
                               scheduler/, or the unet/ dir itself)

Model:
  --model_name_or_path <p>     HF id or local dir (default: $MODEL_NAME_OR_PATH)
  --action_tokenizer_path <p>  FAST tokenizer (default: $ACTION_TOKENIZER_PATH)
  --dataset_type <s>           navsim2va_ross | nuplan2va_ross (default: $DATASET_TYPE)

Training:
  --ngpus <n>                  (8)
  --batch_size <n>             per-device batch, must be >= 2 (default: 2)
  --grad_accum <n>             (1)
  --num_workers <n>            (12)
  --fp <bf16|fp16|fp32>        (bf16)
  --attn_type <sdpa|fa2|eager> (sdpa; fa2 is CUDA-only and needs flash-attn)
  --device <auto|cuda|npu>     (auto)
  --max_steps <n>              (4000)
  --save_steps <n>             (2000)
  --save_total_limit <n>       (2)
  --logging_steps <n>          (10)
  --warmup_steps <n>           (50)
  --learning_rate <f>          (5e-5)
  --weight_decay <f>           (0.1)
  --max_grad_norm <f>          (5.0)
  --model_max_length <n>       (4096)
  --zero_stage <2|3>           (3; maps to scripts/sft/zero{2,3}_offload.json)
  --deepspeed_config <json>    explicit DeepSpeed config (overrides --zero_stage)
  --eval_strategy <s>          (no; this dataset has no eval split)
  --report_to <s>              (tensorboard; also none, wandb)
  --seed <n>                   (42)

Data / loss:
  --max_pixels <n>             (150528 -> 180 image tokens for 16:9)
  --min_pixels <n>             (12544)
  --future_nums <n>            (8)
  --cur_frame_idx <n>          (3; window is -3..+8)
  --action_dim <n>             (3)
  --use_previous_actions <b>   (false)
  --tune_mm_llm <b>            (true)
  --tune_mm_mlp <b>            (true)
  --tune_mm_vision <b>         (true)

Misc:
  --output_dir <dir>           (logs/train_qwen_vla_<timestamp>)
  --exp_name <s>               (navsim)
  --master_port <n>            (23458)
  --shuffle <b>                (true)
  --deterministic              strict reproducibility (slow)
  --log_data_hash              log a SHA256 per batch to <run>/data_hashes/
  -h, --help                   this message
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)    MODEL_NAME_OR_PATH="$2";      shift 2 ;;
    --sd_model_path)         SD_MODEL_PATH="$2";           shift 2 ;;
    --action_tokenizer_path) ACTION_TOKENIZER_PATH="$2";   shift 2 ;;
    --data_path)             DATA_PATH="$2";               shift 2 ;;
    --data_root)             DATA_ROOT="$2";               shift 2 ;;
    --dataset_type)          DATASET_TYPE="$2";            shift 2 ;;
    --output_dir)            OUTPUT_DIR="$2";              shift 2 ;;
    --exp_name)              EXP_NAME="$2";                shift 2 ;;
    --ngpus)                 NGPUS="$2";                   shift 2 ;;
    --master_port)           MASTER_PORT="$2";             shift 2 ;;
    --batch_size)            BATCH_SIZE="$2";              shift 2 ;;
    --grad_accum)            GRAD_ACCUM="$2";              shift 2 ;;
    --num_workers)           NUM_WORKERS="$2";             shift 2 ;;
    --fp)                    FP="$2";                      shift 2 ;;
    --attn_type)             ATTN_TYPE="$2";               shift 2 ;;
    --device)                DEVICE="$2";                  shift 2 ;;
    --max_steps)             MAX_STEPS="$2";               shift 2 ;;
    --save_steps)            SAVE_STEPS="$2";              shift 2 ;;
    --save_total_limit)      SAVE_TOTAL_LIMIT="$2";        shift 2 ;;
    --logging_steps)         LOGGING_STEPS="$2";           shift 2 ;;
    --warmup_steps)          WARMUP_STEPS="$2";            shift 2 ;;
    --learning_rate)         LR="$2";                      shift 2 ;;
    --weight_decay)          WEIGHT_DECAY="$2";            shift 2 ;;
    --max_grad_norm)         MAX_GRAD_NORM="$2";           shift 2 ;;
    --model_max_length)      MODEL_MAX_LENGTH="$2";        shift 2 ;;
    --zero_stage)            ZERO_STAGE="$2";              shift 2 ;;
    --deepspeed_config)      DEEPSPEED_CONFIG="$2"; DEEPSPEED_CONFIG_EXPLICIT=true; shift 2 ;;
    --eval_strategy)         EVAL_STRATEGY="$2";           shift 2 ;;
    --report_to)             REPORT_TO="$2";               shift 2 ;;
    --seed)                  SEED="$2";                    shift 2 ;;
    --max_pixels)            MAX_PIXELS="$2";              shift 2 ;;
    --min_pixels)            MIN_PIXELS="$2";              shift 2 ;;
    --future_nums)           FUTURE_NUMS="$2";             shift 2 ;;
    --cur_frame_idx)         CUR_FRAME_IDX="$2";           shift 2 ;;
    --action_dim)            ACTION_DIM="$2";              shift 2 ;;
    --use_previous_actions)  USE_PREVIOUS_ACTIONS="$2";    shift 2 ;;
    --tune_mm_llm)           TUNE_MM_LLM="$2";             shift 2 ;;
    --tune_mm_mlp)           TUNE_MM_MLP="$2";             shift 2 ;;
    --tune_mm_vision)        TUNE_MM_VISION="$2";          shift 2 ;;
    --shuffle)               SHUFFLE="$2";                 shift 2 ;;
    --deterministic)         DETERMINISTIC=true;           shift ;;
    --log_data_hash)         LOG_DATA_HASH=true;           shift ;;
    --help|-h)               usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

# ============================================================
# Validate inputs
# ============================================================
[ -n "$DATA_PATH" ]     || die "--data_path is required"
[ -n "$DATA_ROOT" ]     || die "--data_root is required"
[ -n "$SD_MODEL_PATH" ] || die "--sd_model_path is required"

[ -f "$DATA_PATH" ]            || die "data pickle not found: $DATA_PATH"
[ -d "$DATA_ROOT" ]            || die "data_root not found: $DATA_ROOT"
[ -e "$ACTION_TOKENIZER_PATH" ] || die "action tokenizer not found: $ACTION_TOKENIZER_PATH"
[ -d "$ROOT/reference/transformers" ] || die "missing vendored transformers: $ROOT/reference/transformers"

# SD1.5: accept either the parent dir or the unet dir.
if [ -f "$SD_MODEL_PATH/config.json" ]; then
  SD_UNET="$SD_MODEL_PATH"
elif [ -f "$SD_MODEL_PATH/unet/config.json" ]; then
  SD_UNET="$SD_MODEL_PATH/unet"
else
  die "no unet/config.json under $SD_MODEL_PATH (pass the SD1.5 parent dir or the unet dir)"
fi
SD_PARENT=$(dirname "$SD_UNET")
[ -f "$SD_PARENT/vae/config.json" ]                    || die "missing $SD_PARENT/vae/config.json"
[ -f "$SD_PARENT/scheduler/scheduler_config.json" ]    || die "missing $SD_PARENT/scheduler/scheduler_config.json"

# DeepSpeed config
if [ "$DEEPSPEED_CONFIG_EXPLICIT" = false ]; then
  case "$ZERO_STAGE" in
    2) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero2_offload.json" ;;
    3) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json" ;;
    *) die "--zero_stage must be 2 or 3 (got '$ZERO_STAGE')" ;;
  esac
fi
[ -f "$DEEPSPEED_CONFIG" ] || die "DeepSpeed config not found: $DEEPSPEED_CONFIG"

# Local model path? Otherwise assume an HF id.
case "$MODEL_NAME_OR_PATH" in
  /*|./*|../*) [ -e "$MODEL_NAME_OR_PATH" ] || die "model path not found: $MODEL_NAME_OR_PATH" ;;
esac

# The ROSS forward does image_hidden[...,0].squeeze(), which collapses the
# batch dim when per-device batch == 1 (modeling_qwen2_5_vl_ross.py:339).
if [ "$BATCH_SIZE" -lt 2 ]; then
  echo "WARNING: --batch_size < 2 currently crashes in the ROSS forward" >&2
  echo "         (squeeze() removes the batch dim). Use 2+ or fix that line." >&2
fi

# ============================================================
# Device / attention setup
# ============================================================
case "$DEVICE" in
  auto|cuda|npu) ;;
  *) die "--device must be auto, cuda, or npu (got '$DEVICE')" ;;
esac

if [ "$DEVICE" = "npu" ] && [ "$ATTN_TYPE" = "fa2" ]; then
  die "--attn_type fa2 is CUDA-only; use sdpa"
fi

if [ "$DEVICE" = "npu" ]; then
  export INF_NAN_MODE_ENABLE=1
  export HCCL_DETERMINISTIC=TRUE
  : "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
elif [ "$DEVICE" = "cuda" ]; then
  export NCCL_DETERMINISTIC=TRUE
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

if [ "$DEVICE" != "npu" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# train_qwen_vla.py reads DEVICE from the environment
export DEVICE

# `reference.transformers...` needs the repo root; qwenvl needs the finetune root.
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================
# Preflight
# ============================================================
python - <<'PY' || exit 1
import importlib.util, sys
import transformers

if not hasattr(transformers, "Qwen2_5_VLConfig"):
    sys.exit(
        f"ERROR: transformers {transformers.__version__} has no Qwen2.5-VL "
        f"({transformers.__file__}).\n"
        "       Install the vendored checkout: pip install -e ./reference/transformers"
    )
print(f"[env] transformers {transformers.__version__} ({transformers.__file__})")

if importlib.util.find_spec("flash_attn") is None:
    print(
        "[warn] flash_attn not installed: qwenvl/train/trainer.py imports it "
        "unconditionally at HEAD (line 7); guard that import for NPU.",
        file=sys.stderr,
    )
PY

case "$FP" in
  bf16) FP_FLAGS="--bf16 True --fp16 False" ;;
  fp16) FP_FLAGS="--bf16 False --fp16 True" ;;
  fp32) FP_FLAGS="--bf16 False --fp16 False" ;;
  *) die "--fp must be bf16, fp16, or fp32 (got '$FP')" ;;
esac

DET_FLAG=""; [ "$DETERMINISTIC" = true ] && DET_FLAG="--deterministic"
HASH_FLAG=""; [ "$LOG_DATA_HASH" = true ] && HASH_FLAG="--log_data_hash"

RUN_DIR="$OUTPUT_DIR/$EXP_NAME"
mkdir -p "$RUN_DIR"
{
  echo "$0 $*"
  echo "Launch time: $(date)"
  echo "Host: $(hostname)"
} > "$RUN_DIR/startup_command.txt"

# ============================================================
# Print config
# ============================================================
echo "=== Qwen VLA training ==="
echo "  model:                 $MODEL_NAME_OR_PATH"
echo "  sd unet:               $SD_UNET"
echo "  data_path:             $DATA_PATH"
echo "  data_root:             $DATA_ROOT"
echo "  action_tokenizer:      $ACTION_TOKENIZER_PATH"
echo "  output:                $RUN_DIR"
echo "  device / attn:         $DEVICE / $ATTN_TYPE"
echo "  precision:             $FP"
echo "  gpus x batch x accum:  ${NGPUS} x ${BATCH_SIZE} x ${GRAD_ACCUM}"
echo "  max_steps / lr:        $MAX_STEPS / $LR"
echo "  max_pixels:            $MAX_PIXELS  (target grid 10x18 = 180 tokens)"
echo "  cur_frame_idx / fut:   $CUR_FRAME_IDX / $FUTURE_NUMS"
echo "  deepspeed:             $DEEPSPEED_CONFIG (zero stage $ZERO_STAGE)"
echo ""

# ============================================================
# Launch
# ============================================================
torchrun \
  --nproc_per_node="$NGPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  "$ROOT/utils/train_qwen_vla.py" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --sd_model_path "$SD_UNET" \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --data_root "$DATA_ROOT" \
    --action_tokenizer_path "$ACTION_TOKENIZER_PATH" \
    --actions_format fast \
    --action_dim "$ACTION_DIM" \
    --cur_frame_idx "$CUR_FRAME_IDX" \
    --future_nums "$FUTURE_NUMS" \
    --use_previous_actions "$USE_PREVIOUS_ACTIONS" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --tune_mm_llm "$TUNE_MM_LLM" \
    --tune_mm_mlp "$TUNE_MM_MLP" \
    --tune_mm_vision "$TUNE_MM_VISION" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --output_dir "$RUN_DIR" \
    $FP_FLAGS \
    --tf32 False \
    --attn_type "$ATTN_TYPE" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing True \
    --dataloader_num_workers "$NUM_WORKERS" \
    --dataloader_shuffle "$SHUFFLE" \
    --seed "$SEED" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --eval_strategy "$EVAL_STRATEGY" \
    --logging_steps "$LOGGING_STEPS" \
    --report_to "$REPORT_TO" \
    $DET_FLAG $HASH_FLAG

echo ""
echo "Training complete. Run dir: $RUN_DIR"
