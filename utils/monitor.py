"""
DriveVLA-W0 训练监控增强模块

功能：
1. HBM显存占用采集（torch_npu接口）
2. MFU精确计算（forward hook统计FLOPs）
3. 通过环境变量控制开关，不影响现有训练

环境变量：
    DRIVEVLA_MONITOR_HBM=1     启用HBM显存采集
    DRIVEVLA_MFU_FLAG=1        启用MFU精确计算（forward hook统计FLOPs）
    DRIVEVLA_PEAK_TFLOPS=486.354  单卡BF16峰值算力（TFLOPS）

使用方式：
    在train_moe.py的LoggingTrainer中初始化
"""

import os
import torch
import torch_npu

try:
    import torch_npu
except ImportError:
    torch_npu = None


# ============================================================
# 1. HBM显存监控
# ============================================================

class HbmMonitor:
    """通过torch.npu接口采集HBM显存占用"""

    def __init__(self):
        self.enabled = os.getenv("DRIVEVLA_MONITOR_HBM") == "1"
        self._records = []

    def snapshot(self, tag=""):
        """采集当前HBM显存快照"""
        if not self.enabled:
            return None
        if torch_npu is None or not torch.npu.is_available():
            return None
        device = torch.device("npu", torch.npu.current_device())
        allocated = torch.npu.memory_allocated(device) / 1e9  # GB
        reserved = torch.npu.memory_reserved(device) / 1e9
        max_allocated = torch.npu.max_memory_allocated(device) / 1e9
        record = {
            "tag": tag,
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "max_allocated_gb": round(max_allocated, 3),
        }
        self._records.append(record)
        return record

    def reset_peak(self):
        """重置峰值统计"""
        if self.enabled and torch_npu is not None and torch.npu.is_available():
            torch.npu.reset_peak_memory_stats(torch.npu.current_device())

    def get_records(self):
        return self._records

    def print_summary(self):
        if not self._records:
            return
        print(f"[HBM] {len(self._records)} snapshots:")
        for r in self._records:
            print(f"  {r['tag']}: alloc={r['allocated_gb']}GB, "
                  f"reserved={r['reserved_gb']}GB, peak={r['max_allocated_gb']}GB")


# ============================================================
# 2. MFU精确计算（forward hook统计FLOPs）
# ============================================================

class FlopsCounter:
    """通过forward hook统计nn.Linear的FLOPs

    口径参考openwam:
      model_FLOPs = 3 × forward(可训练) + 1 × forward(冻结)
      MFU = model_FLOPs / (单步耗时 × 卡数 × 单卡峰值)

    不计入attention score matmul（只统计nn.Linear）
    gradient_checkpointing去重：同一module在同一step内只统计1次forward
    """

    def __init__(self):
        self.enabled = os.getenv("DRIVEVLA_MFU_FLAG") == "1"
        self._hooks = []
        self.forward_flops_trainable = 0
        self.forward_flops_frozen = 0
        self._extra_flops_trainable = 0
        self._peak_tflops = float(os.getenv("DRIVEVLA_PEAK_TFLOPS", "486.354"))
        self._seen_modules = set()
        self._fused_gateup = os.getenv("EMU3_FUSED_GATEUP", "0") == "1"
        self._fused_qkv = os.getenv("EMU3_FUSED_QKV", "0") == "1" and os.getenv("EMU3_ATTN_BSH", "0") != "1"
        self._lm_head_not_hooked = os.getenv("EMU3_LMHEAD_HOOKED", "0") != "1"

    def _add(self, module, flops):
        w = getattr(module, "weight", None)
        if w is not None and w.requires_grad:
            self.forward_flops_trainable += flops
        else:
            self.forward_flops_frozen += flops

    def _linear_hook(self, module, inputs, output):
        mid = id(module)
        if mid in self._seen_modules:
            return
        self._seen_modules.add(mid)
        rows = inputs[0].numel() // module.in_features
        self._add(module, 2.0 * rows * module.in_features * module.out_features)

    def attach(self, model):
        if not self.enabled:
            return
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                h = m.register_forward_hook(self._linear_hook)
                self._hooks.append(h)
        print(f"[FLOPS] Attached hooks to {len(self._hooks)} Linear layers")

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self.forward_flops_trainable = 0
        self.forward_flops_frozen = 0
        self._extra_flops_trainable = 0
        self._seen_modules.clear()

    def add_extra_flops(self, model_config, tokens_per_step, labels_numel=None):
        """Add analytical FLOPs for fused ops that bypass nn.Linear hooks."""
        if not self.enabled:
            return
        H = model_config.hidden_size
        I = model_config.intermediate_size
        V = model_config.vocab_size
        L = model_config.num_hidden_layers
        head_dim = H // model_config.num_attention_heads
        q_dim = model_config.num_attention_heads * head_dim
        kv_dim = model_config.num_key_value_heads * head_dim
        rows = tokens_per_step
        rows_ce = labels_numel if labels_numel else tokens_per_step - 32
        extra = 0
        if self._fused_gateup:
            extra += 2.0 * rows * H * (2 * I) * L
        if self._fused_qkv:
            extra += 2.0 * rows * H * (q_dim + 2 * kv_dim) * L
        if self._lm_head_not_hooked:
            extra += 2.0 * rows_ce * H * V
        self._extra_flops_trainable = extra
        if os.getenv("DRIVEVLA_FLOP_DEBUG") == "1":
            pct = extra / (self.forward_flops_trainable + extra + 1e-9) * 100
            print(f"[FLOP_DEBUG] hook={self.forward_flops_trainable:.3e} extra={extra:.3e} ({pct:.1f}%) total={self.forward_flops_trainable+extra:.3e}")

    @property
    def model_training_flops(self):
        return (3.0 * (self.forward_flops_trainable + self._extra_flops_trainable)
                + 1.0 * self.forward_flops_frozen)

    def compute_mfu(self, step_time_sec, num_cards=8):
        """计算MFU

        Args:
            step_time_sec: 单步耗时（秒）
            num_cards: 卡数（默认8）
        Returns:
            (achieved_tflops, mfu_pct)
        """
        if not self.enabled or step_time_sec <= 0:
            return 0.0, 0.0
        flops = self.model_training_flops
        peak = self._peak_tflops * 1e12
        achieved_tflops = flops / (step_time_sec * 1e12)
        mfu = flops / (step_time_sec * peak) * 100.0
        return achieved_tflops, mfu


# ============================================================
# 3. 统一入口
# ============================================================

def build_monitor():
    """构建监控模块，返回(hbm_monitor, flops_counter)"""
    hbm = HbmMonitor()
    flops = FlopsCounter()
    if hbm.enabled or flops.enabled:
        print(f"[MONITOR] HBM={'ON' if hbm.enabled else 'OFF'}, "
              f"FLOPS={'ON' if flops.enabled else 'OFF'}, "
              f"Peak={flops._peak_tflops}TFLOPS")
    return hbm, flops
