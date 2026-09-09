from msprobe.pytorch import seed_all
seed_all(seed=1234, mode=True, rm_dropout=True)

import os
import os.path as osp
import random
import numpy as np
import torch
import gc
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
import hashlib
import json
import torch.distributed as dist
from datetime import datetime
import threading
from queue import Queue
import time
import inspect

# ---------------------------------------------------------------------------
# Device detection: NPU > CUDA > CPU  (override via DEVICE env var)
# ---------------------------------------------------------------------------
_device_override = os.environ.get("DEVICE", "auto")

if _device_override == "npu":
    # When the user explicitly asks for NPU, try importing torch_npu.
    # The bare-except at module level is fine here — we want the ImportError
    # to propagate so the user sees what's wrong.
    import torch_npu  # noqa: F401 — will raise ImportError if missing
    _npu_available = torch.npu.is_available()
    if not _npu_available:
        raise RuntimeError("DEVICE=npu set but no NPU detected (torch.npu.is_available()=False)")
    _device_type = "npu"

elif _device_override == "cuda":
    _device_type = "cuda"

elif _device_override == "cpu":
    _device_type = "cpu"

else:
    # "auto" — import torch_npu quietly; failure is fine, fall back to cuda/cpu
    try:
        import torch_npu  # noqa: F401
        _npu_available = torch.npu.is_available()
    except Exception:
        _npu_available = False
    _device_type = "npu" if _npu_available else ("cuda" if torch.cuda.is_available() else "cpu")


def device_synchronize():
    if _device_type == "npu":
        torch.npu.synchronize()
    elif _device_type == "cuda":
        torch.cuda.synchronize()


def device_empty_cache():
    if _device_type == "npu":
        torch.npu.empty_cache()
    elif _device_type == "cuda":
        torch.cuda.empty_cache()


def device_manual_seed_all(seed: int):
    if _device_type == "npu":
        torch.npu.manual_seed_all(seed)
    elif _device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录(即包含train和reference的目录)
parent_dir = os.path.dirname(current_dir)
# 添加reference/Emu3路径到sys.path
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig
from transformers import AutoModel, Trainer, TrainerCallback
from datasets import Emu3DrivingDataset
from datasets import Emu3DrivingVAVADataset
from datasets import Emu3DrivingNuplan6VADataset
from torch.utils.data import WeightedRandomSampler, DataLoader, RandomSampler, SequentialSampler
from transformers.trainer_utils import has_length

# ---------------------------------------------------------------------------
# Gradient-clip verification (--log_grad_clip)
#
# HF Trainer already writes `grad_norm` (the PRE-clip global gradient norm) into
# each train log entry — for both the plain-torch and DeepSpeed paths. To double
# check that --max_grad_norm actually clamps, we additionally measure the
# POST-clip norm:
#   * torch path: wrap torch.nn.utils.clip_grad_norm_ and re-measure the norm of
#     the clipped grads right after it runs (a real measurement).
#   * DeepSpeed: clipping happens inside the engine's optimizer.step(), which is
#     not readable afterwards, so post = min(pre, max) — exact for global-norm
#     clipping — and grad_clip_measured=False flags it as derived.
# ---------------------------------------------------------------------------
_GCLIP = {}            # most recent clip event: {pre, post, clipped, max}
_CLIP_PATCHED = False


def _install_grad_clip_hook():
    """Wrap torch.nn.utils.clip_grad_norm_ once per process (idempotent).

    Records the pre-clip norm (the function's return value) and, only when
    clipping actually triggered, re-measures the global norm of the clipped
    grads. Stored in module-level _GCLIP, read by LoggingTrainer.log().
    """
    global _CLIP_PATCHED
    if _CLIP_PATCHED:
        return
    _orig = torch.nn.utils.clip_grad_norm_

    def _hook(parameters, max_norm, norm_type=2.0):
        pre = _orig(parameters, max_norm, norm_type=norm_type)
        if not (max_norm and max_norm > 0):
            return pre
        pre_f = float(pre)
        clipped = pre_f > float(max_norm)
        post = None
        if clipped:
            grads = [p.grad.detach() for p in parameters if p.grad is not None]
            if grads:
                with torch.no_grad():
                    post = float((sum(g.float().abs().pow(norm_type).sum() for g in grads)) ** (1.0 / norm_type))
        else:
            post = pre_f  # no clipping applied → post == pre
        _GCLIP.update(pre=pre_f, post=post, clipped=clipped, max=float(max_norm))
        return pre

    torch.nn.utils.clip_grad_norm_ = _hook
    _CLIP_PATCHED = True


class _RawGradNormCallback(TrainerCallback):
    """Report the raw pre-clip global grad norm when --max_grad_norm is disabled.

    HF Trainer only computes `grad_norm` inside the clipping branch, so when
    --max_grad_norm is 0 (clipping off) it logs nothing. This callback measures
    the norm every sync step instead, and _inject_grad_clip_logs writes it under
    the same `grad_norm` key as clipped runs → no-clip tiers stay comparable.

    DeepSpeed computes the true global norm via the engine's collective
    (get_global_grad_norm); non-DeepSpeed falls back to a local L2 over grads.
    """
    def __init__(self, trainer):
        self._trainer = trainer
        self.value = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            return  # HF already logs pre-clip grad_norm in the clipping branch
        self.value = self._global_grad_norm()

    def on_log(self, args, state, control, logs, **kwargs):
        # Consume the value so a later eval log entry can't reuse it.
        self.value = None

    def _global_grad_norm(self):
        # With DeepSpeed, self.model is the raw model; the engine (which owns
        # get_global_grad_norm) is self.model_wrapped / self.deepspeed.
        models = (
            getattr(self._trainer, "model_wrapped", None),
            getattr(self._trainer, "deepspeed", None),
            self._trainer.model,
        )
        for m in models:
            get_global = getattr(m, "get_global_grad_norm", None)
            if get_global is not None:
                try:
                    v = get_global()
                    return float(v) if v is not None else None
                except Exception:
                    return None
        params = [p for p in self._trainer.model.parameters() if p.grad is not None]
        if not params:
            return None
        with torch.no_grad():
            return float(sum(p.grad.float().norm().item() ** 2 for p in params) ** 0.5)


class NpuProfilerCallback(TrainerCallback):
    """Wrap the HF Trainer loop with torch_npu.profiler (NPU-only, opt-in).

    Mirrors a manual `with torch_npu.profiler.profile(...) as prof:` +
    `prof.step()` per train_dataloader batch: start on_train_begin, step once
    per micro-batch (on_step_end), stop on_train_end. Schedule is the
    low-overhead one — skip N steps, then one wait/warmup/active window →
    a single fully-warmed trace.

    level 0 (default): NPU activity only, minimal overhead.
    level 1: CPU+NPU activities, record_shapes=True, profiler_level=Level1 —
    richer trace (op shapes), higher overhead.
    """
    def __init__(self, trace_dir: str, level: int = 0):
        self._trace_dir = trace_dir
        self._level = level
        self._prof = None

    def _build(self):
        if self._level == 1:
            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                with_stack=False,
                record_shapes=True,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=26),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self._trace_dir),
            )
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=26),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self._trace_dir),
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._prof = self._build()
        self._prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self._prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self._prof.stop()


class MemoryEfficientTrainer(tf.Trainer):
    """最简单的显存回收Trainer"""
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        # 评估前清理显存
        device_empty_cache()
        gc.collect()

        # 执行评估
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # 评估后清理显存
        device_empty_cache()
        gc.collect()
        
        return result

class LoggingTrainer(tf.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wall-clock timestamp of the last log() call, for per-step time tracking
        self._last_log_time = None
        # Accumulated token count (non-padding) since last log() call — populated
        # by training_step() from attention_mask.sum() and consumed by log().
        self._tokens_since_last_log = 0

        # Data hash logging for cross-platform (NPU vs GPU) data-order verification
        self._hash_logfile = None
        if getattr(self.args, 'log_data_hash', False):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            hash_dir = os.path.join(self.args.output_dir, 'data_hashes')
            os.makedirs(hash_dir, exist_ok=True)
            hash_path = os.path.join(hash_dir, f'rank{rank}.jsonl')
            self._hash_logfile = open(hash_path, 'w')
            if self.state.is_world_process_zero:
                print(f'[DataHash] Logging batch hashes to {hash_path}')

        # Submodule timing logging for profiling per-component training time.
        # All ranks record — ZeRO-3 all-gather times can differ per rank.
        self._submodule_logfile = None
        self._submodule_last_step = 0
        if getattr(self.args, 'log_submodule_time', False):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            sub_dir = os.path.join(self.args.output_dir, 'submodule_times')
            os.makedirs(sub_dir, exist_ok=True)
            sub_path = os.path.join(sub_dir, f'rank{rank}.jsonl')
            self._submodule_logfile = open(sub_path, 'w')
            if self.state.is_world_process_zero:
                print(f'[SubmoduleTime] Logging submodule times to {sub_dir}/rank*.jsonl')

        self.log_queue = None
        # Only the main process will handle file I/O and the logging thread.
        if self.state.is_world_process_zero:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"sample_log_consolidated_{timestamp}.json"
            self.log_filepath = osp.join(self.args.output_dir, log_filename)

            self.log_queue = Queue()
            self.logging_thread = threading.Thread(target=self._log_writer, daemon=True)
            self.logging_thread.start()

        # Gradient-clip verification (--log_grad_clip): measure pre/post clip
        # norms so trainer_state.json can confirm --max_grad_norm takes effect.
        # The callback also reports the raw pre-clip norm when clipping is
        # disabled (HF skips grad_norm entirely when max_grad_norm <= 0).
        if getattr(self.args, 'log_grad_clip', False):
            _install_grad_clip_hook()
            self._raw_grad_norm_cb = _RawGradNormCallback(self)
            self.add_callback(self._raw_grad_norm_cb)

    def log(self, logs: dict, start_time=None) -> None:
        self._inject_grad_clip_logs(logs)
        # Wall-clock seconds since the previous log() call, written into each
        # trainer_state.json log_history entry as `time_elapsed`. With
        # logging_steps=1 (alignment experiments) this is the time per step.
        now = time.time()
        if self._last_log_time is not None:
            logs["time_elapsed"] = round(now - self._last_log_time, 3)
            logs["tokens_per_step"] = self._tokens_since_last_log
            # tokens/sec = actual non-padding tokens (from attention_mask) / time_elapsed
            # Multiply by world_size for global throughput, consistent with the
            # old bs × ws × sl formula.
            if self._tokens_since_last_log > 0 and logs["time_elapsed"] > 0:
                ws = getattr(self.args, "world_size", 1)
                logs["tokens_per_sec"] = round(self._tokens_since_last_log * ws / logs["time_elapsed"], 1)
        self._last_log_time = now
        self._tokens_since_last_log = 0
        # Older installed transformers: Trainer.log(logs) only; 4.56+ also accepts start_time
        if start_time is not None and "start_time" in inspect.signature(Trainer.log).parameters:
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Collect per-submodule forward times for profiling.
        # All ranks record independently — ZeRO-3 all-gather and rank-0
        # overhead (logging, callbacks) can cause per-rank timing variance.
        if getattr(self.args, 'log_submodule_time', False):
            model = self.model.module if hasattr(self.model, 'module') else self.model
            record = {'step': int(self.state.global_step),
                      'steps': int(self.state.global_step - self._submodule_last_step)}
            for name, module in model.named_modules():
                st = getattr(module, '_submodule_times', None)
                if st:
                    prefix = f'{name}.' if name else ''
                    for k, v in st.items():
                        record[f'{prefix}{k}'] = round(v, 6)
            self._submodule_logfile.write(json.dumps(record) + '\n')
            self._submodule_logfile.flush()
            self._submodule_last_step = int(self.state.global_step)
            model.reset_submodule_times()

    def _inject_grad_clip_logs(self, logs: dict) -> None:
        """Add pre/post gradient-clip norms to a train log entry (--log_grad_clip).

        Only fires on steps where HF logged a `grad_norm` (i.e. train steps with
        max_grad_norm > 0 — eval/val log entries have none and are left alone).
        HF's `grad_norm` is the pre-clip global norm in both the torch and
        DeepSpeed paths. The post-clip value is measured by the clip hook when it
        fired for this step (pre matches); DeepSpeed clips inside optimizer.step()
        so its post value is min(pre, max), flagged via grad_clip_measured=False.
        """
        if not getattr(self.args, 'log_grad_clip', False):
            return
        max_norm = getattr(self.args, "max_grad_norm", None)
        if max_norm is None or max_norm <= 0:
            # Clipping disabled: HF won't log grad_norm at all. Inject the raw
            # pre-clip norm captured by _RawGradNormCallback, under the same
            # `grad_norm` key as clipped runs so tiers stay comparable.
            cb = getattr(self, "_raw_grad_norm_cb", None)
            if cb is not None and cb.value is not None:
                logs["grad_norm"] = round(float(cb.value), 4)
            return
        pre = logs.get("grad_norm")
        if pre is None:
            return
        pre = float(pre)
        max_f = float(max_norm)
        # The hook's post value only belongs to this step if its pre matches HF's.
        measured = _GCLIP.get("pre") is not None and abs(_GCLIP["pre"] - pre) < 1e-3
        post = _GCLIP.get("post") if measured and _GCLIP.get("post") is not None else min(pre, max_f)
        logs["grad_norm_before_clip"] = round(pre, 4)
        logs["grad_norm_after_clip"] = round(float(post), 4)
        logs["grad_clipped"] = bool(pre > max_f)
        logs["grad_clip_measured"] = measured

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_length:
            return super()._get_train_sampler(train_dataset)

        if self.args.dataloader_shuffle:
            return RandomSampler(train_dataset)
        else:
            return SequentialSampler(train_dataset)

    def _log_writer(self):
        log_data = {}
        with open(self.log_filepath, 'w') as f:
            while True:
                # Blocks until an item is available
                data = self.log_queue.get()
                
                # A sentinel value (None) indicates the end of training
                if data is None:
                    break
                
                step, indices = data
                log_data[step] = indices
                
                # Overwrite the file with the updated data at each step
                f.seek(0)
                json.dump(log_data, f, indent=4)
                f.truncate()
                f.flush() # Ensure data is written to disk

    # Non-model keys added by VAVA/AR datasets that Emu3MoE.forward() doesn't accept.
    # Pop in both training_step and compute_loss so evaluation doesn't crash on them.
    _VAVA_EXTRA_KEYS = {"pre_action", "token", "vlm_input_ids", "vlm_attention_mask", "vlm_labels", "action_input_ids"}

    def _strip_non_model_keys(self, inputs: dict) -> dict:
        """Remove dataset keys that aren't accepted by model.forward().

        Mutates inputs in place and returns it for convenience.
        """
        inputs.pop("index", None)
        inputs.pop("cmd", None)
        for k in self._VAVA_EXTRA_KEYS:
            inputs.pop(k, None)
        return inputs

    def training_step(self, model: torch.nn.Module, inputs: dict) -> torch.Tensor:
        # Data hash logging -- compute BEFORE forward (the model can mutate inputs)
        if self._hash_logfile is not None:
            step = self.state.global_step
            batch_hash = compute_batch_hash(inputs)
            record = {'step': step, 'hash': batch_hash}
            self._hash_logfile.write(json.dumps(record) + '\n')
            self._hash_logfile.flush()  # survive crashes

        # Pop the index first, since it's not a model input.
        indices = inputs.pop("index", None)
        # Pop VAVA-only keys that the model forward doesn't accept
        self._strip_non_model_keys(inputs)

        # Count non-padding tokens in this batch for TPS logging.
        # Use input_ids.numel() rather than attention_mask.sum() because TPS
        # is a compute-cost metric: padding tokens still consume FLOPs and
        # memory even though they're masked from the loss.
        ids = inputs.get("input_ids")
        if ids is not None:
            self._tokens_since_last_log += int(ids.numel())

        loss = super().training_step(model, inputs)

        # ponytail: flush device stream so DeepSpeed get_global_grad_norm()
        # returns the actual gradient (not 0.0) on all platforms. Without this,
        # CUDA reports 3 zero-grad_norm steps vs NPU's 2 due to async allreduce.
        if self.state.global_step < 5:
            device_synchronize()

        # Only proceed with logging if we have indices and are in training mode
        if indices is not None and self.is_in_train:
            if dist.is_initialized():
                # 🔥 CRITICAL: ALL processes must participate in this collective operation
                gathered_indices_list = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_indices_list, indices.cpu().tolist())
                
                # Only the main process will log the consolidated data
                if self.state.is_world_process_zero:
                    # Flatten the list of lists into a single list
                    consolidated_indices = [item for sublist in gathered_indices_list for item in sublist]
                    # Put the data into the queue for the logging thread to process.
                    self.log_queue.put((self.state.global_step, consolidated_indices))
            else:
                # Non-distributed case: only main process logs
                if self.state.is_world_process_zero:
                    consolidated_indices = indices.cpu().tolist()
                    self.log_queue.put((self.state.global_step, consolidated_indices))

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """Strip non-model keys before model(**inputs) — mirrors training_step.

        Without this, evaluation (which calls compute_loss → model(**inputs) via
        prediction_step) crashes with ``unexpected keyword argument 'pre_action'``
        when the dataset adds VAVA / AR auxiliary keys.
        """
        self._strip_non_model_keys(inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def __del__(self):
        # Gracefully shut down the logging thread
        if self.state.is_world_process_zero and hasattr(self, 'log_queue') and self.log_queue is not None:
            # Signal the logging thread to terminate
            self.log_queue.put(None)
            # Wait for the logging thread to finish its work
            self.logging_thread.join()


class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        # 从 train_dataset 中获取 sample_weights
        # NPU does not support float64; use float32 everywhere
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.float32
        )
        # 用 sample_weights 构建 WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Strip non-model keys before model(**inputs).

        Same as LoggingTrainer.compute_loss: VAVA/AR datasets add pre_action,
        cmd, token, etc. that Emu3MoE.forward() doesn't accept.
        """
        inputs.pop("index", None)
        inputs.pop("cmd", None)
        for k in ({"pre_action", "token", "vlm_input_ids", "vlm_attention_mask",
                   "vlm_labels", "action_input_ids"}):
            inputs.pop(k, None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    driving: bool = field(default=False)
    use_previous_actions: bool = field(default=False)
    use_flip: bool = field(default=False)
    cur_frame_idx: int = field(default=3)
    data_type: str = field(default="navsim") 
    vq_root: str = field(default="/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/nuplan/processed_data/vq_codes")
    pre_action_frames: int = field(default=3)
    resolution: str = field(default="36,64")
    action_hz: float = field(default=2)
    va_pair_num: int = field(default=6)
    
@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default=None)  # None → auto: fa2 on CUDA (fallback sdpa), sdpa on NPU/CPU
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)
    dataloader_shuffle: bool = field(default=True)  # False → deterministic data order (NPU/GPU alignment)
    evaluation_strategy: str = field(default="steps")  # or "epoch"
    eval_steps: Optional[int] = field(default=1000)     # 每 1000 step 验证一次
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_accumulation_steps: Optional[int] = field(default=1)
    save_on_each_node: bool = field(default=False)  # 只在主节点保存
    save_only_model: bool = field(default=False)
    deterministic: bool = field(default=False)  # enable strict reproducibility for NPU vs GPU debugging
    log_data_hash: bool = field(default=False)  # log SHA256 hash per batch for cross-platform data verification
    log_submodule_time: bool = field(default=False)  # log per-submodule forward times for profiling
    log_grad_clip: bool = field(default=False)  # log pre/post gradient-clip norms to verify --max_grad_norm takes effect
    npu_profiling: bool = field(default=False)  # wrap the train loop with torch_npu.profiler (NPU only)
    npu_profiler_level: int = field(default=0)  # 0=NPU only, 1=CPU+NPU + record_shapes + ProfilerLevel.Level1
    no_save_weights: bool = field(default=False)  # skip final save_state/save_model (smoke runs) — save_strategy=no must also be set

def load_model(model_args, model_config, training_args):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    # FA2 is CUDA-only and must be explicitly requested. On non-CUDA an explicit
    # fa2 is an error (no silent fallback); on CUDA we fall back to sdpa if
    # flash-attn isn't installed.
    attn_impl = training_args.attn_type
    if attn_impl in ("fa2", "flash_attention_2"):
        if _device_type != "cuda":
            raise ValueError(
                f"attn_type='{attn_impl}' is CUDA-only (got device={_device_type}); "
                f"use attn_type='sdpa' (the NPU/CPU default)."
            )
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
            print("[Device] flash-attn not installed — falling back to sdpa")

    # Map CLI shorthand to the HF/model implementation name — Emu3's attention
    # classes key on "flash_attention_2", not "fa2".
    attn_impl = {"fa2": "flash_attention_2"}.get(attn_impl, attn_impl)
    print(f"[Device] attention implementation: {attn_impl}")

    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = attn_impl
        model = Emu3MoE(config=model_config)
    else:
        # Suppress DeepSpeed auto-init during model loading — the Trainer
        # will wrap the model with DeepSpeed later.  Without this, ZeRO-3's
        # batch-size assertion fires before the Trainer can set the values.
        _ds_env = os.environ.pop("DS_CONFIG", None)
        _cf_env = os.environ.pop("CONFIG_FILE", None)
        try:
            model = Emu3MoE.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
            )
        finally:
            if _ds_env is not None:
                os.environ["DS_CONFIG"] = _ds_env
            if _cf_env is not None:
                os.environ["CONFIG_FILE"] = _cf_env

    # Training best-practices:
    #   1. Disable KV cache — wasteful during training (no generation),
    #      saves ~2 GiB of peak memory.
    #   2. Gradient checkpointing is handled by the Trainer via the
    #      --gradient_checkpointing flag, but we keep use_cache=False
    #      regardless so the model doesn't allocate unused KV buffers.
    model.config.use_cache = False

    return model

def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    elif data_args.real_robot:
        return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    elif data_args.driving:
        return Emu3DrivingDataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    if data_args.data_type == "navsim_vava":
        full_dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    elif data_args.data_type == "nuplan_6va":
        full_dataset = Emu3DrivingNuplan6VADataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3DrivingDataset(data_args, tokenizer=tokenizer)
        
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def set_reproducibility(seed: int, deterministic: bool = True):
    """Strict reproducibility setup for cross-platform (NPU vs GPU) comparison.

    Mirrors the msprobe.pytorch.seed_all() approach — see
    https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md
    """
    # Python-level hash seed — must be set BEFORE any dict/set operations
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if _device_type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            # NCCL deterministic communication (GPU)
            os.environ.setdefault("NCCL_DETERMINISTIC", "TRUE")
            os.environ.setdefault("NCCL_CROSS_NIC", "1")
        # HCCL (NPU) deterministic communication
        os.environ.setdefault("HCCL_DETERMINISTIC", "TRUE")
        # NPU non-saturation mode: ensure overflow → Inf/NaN (matches GPU default)
        os.environ.setdefault("INF_NAN_MODE_ENABLE", "1")

    return seed  # useful for callers to chain


def disable_model_dropout(model: torch.nn.Module) -> None:
    """Disable all dropout layers in the model for deterministic precision comparison.

    Walks the module tree and sets p=0 on every Dropout/DropoutNd instance.
    Mirrors msprobe.pytorch.seed_all(rm_dropout=True) behavior.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p = 0.0
            print(f"[Reproducibility] Dropout disabled: {name}")


def compute_batch_hash(inputs):
    """Compute a deterministic SHA256 hash of a batch for cross-platform comparison.

    Each tensor is converted float32 -> CPU numpy -> tobytes() for bit-identical
    hashing across NPU and GPU platforms.
    """
    h = hashlib.sha256()
    for key in sorted(inputs.keys()):
        val = inputs[key]
        h.update(key.encode())
        if isinstance(val, torch.Tensor):
            h.update(val.detach().float().cpu().numpy().tobytes())
        elif val is None:
            h.update(b"<NONE>")
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, torch.Tensor):
                    h.update(v.detach().float().cpu().numpy().tobytes())
                else:
                    h.update(str(v).encode())
        else:
            h.update(str(val).encode())
    return h.hexdigest()


def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"[Device] detected: {_device_type}")

    # Auto-resolve attention implementation: FA2 on CUDA (fallback to sdpa in
    # load_model if flash-attn is missing), SDPA on NPU/CPU. An explicit fa2 on
    # NPU errors loudly — no silent fallback. "auto" covers DEVICE=auto shells
    # that defer resolution to the real device.
    if training_args.attn_type in (None, "auto"):
        training_args.attn_type = "fa2" if _device_type == "cuda" else "sdpa"
        print(f"[Device] attn_type auto → {training_args.attn_type}")

    # Strict reproducibility for cross-platform (NPU vs GPU) comparison
    if training_args.deterministic:
        set_reproducibility(training_args.seed, deterministic=True)
        print(f"[Reproducibility] enabled (seed={training_args.seed})")

    # Handle resolution parameter conversion from string to tuple
    if isinstance(data_args.resolution, str):
        print(f"Converting resolution from string '{data_args.resolution}' to tuple")
        data_args.resolution = tuple(map(int, data_args.resolution.split(',')))
        print(f"Resolution after conversion: {data_args.resolution}")

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load model configuration and tokenizer
    model_config = Emu3MoEConfig.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    model_config.log_submodule_time = training_args.log_submodule_time
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize model
    model = load_model(model_args, model_config, training_args)

    if training_args.deterministic:
        disable_model_dropout(model)

    # Initialize dataset
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)
    # train_dataset = get_dataset(data_args, tokenizer)

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, 
            tokenizer=tokenizer,
        )
    else:
        # Setup Trainer
        trainer = LoggingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # ✅ 加上这个
            tokenizer=tokenizer,
        )

    # NPU-only profiling (opt-in via --npu_profiling). Trace dir under
    # output_dir so a run's traces live with the run.
    if training_args.npu_profiling:
        if _device_type == "npu":
            lvl = training_args.npu_profiler_level
            trace_dir = osp.join(training_args.output_dir, "npu_profile" if lvl == 0 else f"npu_profile_level{lvl}")
            trainer.add_callback(NpuProfilerCallback(trace_dir, lvl))
            print(f"[NpuProfiler] torch_npu profiler level {lvl} active — traces → {trace_dir}")
        else:
            print(f"[NpuProfiler] --npu_profiling requires NPU (got {_device_type}) — ignored")


    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save training state (always) and model weights (skipped for smoke runs)
    trainer.save_state()
    device_synchronize()
    if not training_args.no_save_weights:
        trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    gc.disable()  # kill GC-pause jitter in step timings; refcounting still frees memory
    train()
