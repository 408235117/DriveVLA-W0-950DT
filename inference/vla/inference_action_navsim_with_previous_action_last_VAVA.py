import os, sys, yaml, json, pickle, argparse
import torch
import torch.distributed as dist
import numpy as np
from scipy.fft import idct
from transformers import (
    AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor, LogitsProcessor
)
from emu3.mllm import Emu3MoE, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor
from tqdm import tqdm


def robust_decode_action(action_tokenizer, token_list, time_horizon, action_dim):
    """
    Decode action tokens with padding for short sequences.
    Fallback to manual decode if the tokenizer's built-in decode fails.
    """
    try:
        result = action_tokenizer.decode(token_list, time_horizon=time_horizon, action_dim=action_dim)
        # Check if result is all zeros (decode failure fallback)
        if not np.allclose(result, 0):
            return result
    except Exception:
        pass

    # Manual decode with padding
    expected_elems = time_horizon * action_dim
    decoded_actions = []
    for token in token_list:
        try:
            decoded_str = action_tokenizer.bpe_tokenizer.decode(token)
            coeff_flat = np.array(list(map(ord, decoded_str)), dtype=np.int64) + action_tokenizer.min_token
            n = int(coeff_flat.size)
            if n < expected_elems:
                coeff_flat = np.pad(coeff_flat, (0, expected_elems - n), mode="constant", constant_values=0)
            elif n > expected_elems:
                coeff_flat = coeff_flat[:expected_elems]
            decoded_dct_coeff = coeff_flat.reshape(time_horizon, action_dim)
        except Exception:
            decoded_dct_coeff = np.zeros((time_horizon, action_dim))
        decoded_actions.append(idct(decoded_dct_coeff / action_tokenizer.scale, axis=0, norm="ortho"))
    return np.stack(decoded_actions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emu_hub", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--train_meta_pkl", required=True, type=str)
    parser.add_argument("--input_num_frame", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_action_tokens", type=int, default=0)
    parser.add_argument("--norm_config", type=str, default="configs/normalizer_navsim_trainval/norm_stats.json")
    parser.add_argument("--vq_hub", type=str, default="/data/models/Emu3-Stage1")
    parser.add_argument("--vision_hub", type=str, default="/data/models/Emu3-VisionTokenizer")
    parser.add_argument("--fast_tokenizer", type=str, default="/data/models/physical-intelligence-fast")
    return parser.parse_args()


def setup_distributed():
    _device_type = os.environ.get("DEVICE", "auto")
    if _device_type == "npu":
        backend = "hccl"
    elif _device_type == "cuda":
        backend = "nccl"
    else:
        # auto-detect
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                backend = "hccl"
            else:
                backend = "nccl"
        except Exception:
            backend = "nccl"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids
    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if scores.ndim == 2:
            mask[:, self.allowed_token_ids] = True
        else:
            mask[self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


class SuppressEOSForNStepsLogitsProcessor(LogitsProcessor):
    """Suppress EOS token for the first `min_steps` generation steps."""
    def __init__(self, eos_token_id, min_steps):
        self.eos_token_id = eos_token_id
        self.min_steps = min_steps
        self._step = 0

    def __call__(self, input_ids, scores):
        if self._step < self.min_steps:
            if scores.ndim == 2:
                scores[:, self.eos_token_id] = -float("inf")
            else:
                scores[self.eos_token_id] = -float("inf")
        self._step += 1
        return scores

def wrap_action_sequence(tokenizer, action_ids) -> torch.Tensor:
    """
    Wraps a sequence of action token IDs with special tokens (beginning and end).

    Args:
        action_ids (List[int]): The sequence of action token IDs.

    Returns:
        torch.Tensor: A tensor containing the wrapped sequence.
    """
    # Encode the beginning and end action tokens
    action_begin = tokenizer.encode(tokenizer.boa_token)[0]
    action_end = tokenizer.encode(tokenizer.eoa_token)[0]

    # Wrap the action sequence
    wrapped_action = [action_begin] + action_ids + [action_end]
    
    # Convert to a PyTorch tensor
    return torch.tensor(wrapped_action, dtype=torch.long)

def main():
    args = parse_args()

    CONFIG = {
        "emu_hub": args.emu_hub,
        "output_dir": args.output_dir,
        "train_meta_pkl": args.train_meta_pkl,

        "vq_hub": args.vq_hub,
        "vision_hub": args.vision_hub,
        "fast_tokenizer": args.fast_tokenizer,
        "norm_config": args.norm_config,
        "token_yaml": "inference/navsim/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
    }

    action_predict_frame = 8
    num_frames = args.input_num_frame
    DEBUG = True

    rank, world_size = setup_distributed()

    # Device-agnostic device string
    _device_type = os.environ.get("DEVICE", "auto")
    if _device_type == "npu":
        device = f"npu:{rank}"
    elif _device_type == "cuda":
        device = f"cuda:{rank}"
    else:
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                device = f"npu:{rank}"
            else:
                device = f"cuda:{rank}"
        except Exception:
            device = f"cuda:{rank}"

    with open(CONFIG["train_meta_pkl"], 'rb') as f:
        train_meta = pickle.load(f)

    with open(CONFIG["token_yaml"], 'r') as f:
        token_list = yaml.safe_load(f)['tokens']

    norm_cfg = json.load(open(CONFIG["norm_config"], 'r'))
    action_low = torch.tensor(norm_cfg['norm_stats']['libero']['q01']).to(device)
    action_high = torch.tensor(norm_cfg['norm_stats']['libero']['q99']).to(device)

    # FA2 is CUDA-only; fall back to sdpa on NPU / other devices, or if not installed
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2" if ("cuda" in device) else "sdpa"
    except ImportError:
        _attn_impl = "sdpa"
    model = Emu3MoE.from_pretrained(CONFIG["emu_hub"],
        torch_dtype=torch.bfloat16,
        attn_implementation=_attn_impl,
        trust_remote_code=True,
    ).to(device).eval()

    GENERATION_CONFIG = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=151845,
        do_sample=False,
    )

    tokenizer = Emu3Tokenizer.from_pretrained(CONFIG["vq_hub"], use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(CONFIG["vision_hub"], trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(CONFIG["vision_hub"], trust_remote_code=True).to(device).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    action_tokenizer = AutoProcessor.from_pretrained(CONFIG["fast_tokenizer"], trust_remote_code=True)

    last_token_id = tokenizer.pad_token_id - 1
    allowed_token_ids = list(range(last_token_id - action_tokenizer.vocab_size, last_token_id + 1)) + [151845]
    kwargs = dict(mode='VLA', padding="longest")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 为当前 rank 分配的 index 子集
    total_tasks = list(range(rank, len(train_meta), world_size))
    if args.max_samples is not None:
        total_tasks = total_tasks[:args.max_samples]  # truncate per rank
    pbar = tqdm(total=len(total_tasks), desc=f"Rank {rank}", position=rank)
    # token_length = []
    for local_idx, idx in enumerate(total_tasks):
        task_data = train_meta[idx]
        token_name = token_list[idx]

        image_list = task_data['image']
        action_list = task_data['action']
        text = task_data['text']
        cur_idx = 3

        # If pickle paths differ from actual data location, set DATA_PATH_MAP="old_prefix|new_prefix"
        _path_map = os.environ.get("DATA_PATH_MAP", "").split("|")
        def _remap(p):
            return p.replace(_path_map[0], _path_map[1]) if len(_path_map) >= 2 else p

        video_code = [
            torch.from_numpy(np.load(_remap(img)))
            for img in image_list[cur_idx-2*(num_frames-1):cur_idx+1:2]
        ]
        video_code = torch.stack(video_code, dim=1).to(device)
        input_text = text[cur_idx]

        pos_inputs = processor.video_process(
            text=input_text, video_tokens=video_code, gripper_tokens=None,
            context_frames=num_frames, frames=1, return_tensors="pt", **kwargs
        )
        # remove bos in pos_inputs
        pos_inputs['input_ids'] = pos_inputs.input_ids[:, 1:]
        pos_inputs['attention_mask'] = pos_inputs.attention_mask[:, 1:]
        pos_inputs['token_type_ids'] = pos_inputs.token_type_ids[:, 1:]


        pre_image_list = task_data['pre_1s_image']
        pre_action_list = task_data['pre_1s_action']
        pre_text = task_data['pre_1s_text']

        pre_video_code = [
            torch.from_numpy(np.load(_remap(img)).reshape(1, 18, 32))
            for img in pre_image_list[cur_idx-2*(num_frames-1):cur_idx+1:2]
        ]
        pre_video_code = torch.stack(pre_video_code, dim=1).to(device)
        pre_input_text = pre_text[cur_idx]

        pre_pos_inputs = processor.video_process(
            text=pre_input_text, video_tokens=pre_video_code, gripper_tokens=None,
            context_frames=num_frames, frames=1, return_tensors="pt", **kwargs
        )


        use_previous_actions = True
        if use_previous_actions:
            # append previous actions to the previous index input
            pre_previous_actions = np.array(pre_action_list[0:cur_idx])
            pre_previous_action_ids = action_tokenizer(pre_previous_actions)[0]
            pre_previous_action_ids = [last_token_id - id for id in pre_previous_action_ids]

            # pre_previous_action_sample = wrap_action_sequence(tokenizer, pre_previous_action_ids).unsqueeze(0)
            pre_previous_action_sample = torch.tensor(pre_previous_action_ids).unsqueeze(0)
            pre_previous_action_mask = torch.ones_like(pre_previous_action_sample, dtype=torch.long)
            #remove boa to the end
            pre_pos_inputs.input_ids = torch.cat([pre_pos_inputs.input_ids[:,:-1], pre_previous_action_sample, pre_pos_inputs.input_ids[:,-1:]], dim=-1)
            pre_pos_inputs.attention_mask = torch.cat([pre_previous_action_mask, pre_pos_inputs.attention_mask], dim=-1)


            # append selected action to the previous index input and the first action just need two waypoints
            pre_selected_actions = np.array(pre_action_list[cur_idx:cur_idx+2])
            pre_selected_action_ids = action_tokenizer(pre_selected_actions)[0]
            pre_selected_action_ids = [last_token_id - id for id in pre_selected_action_ids]
            # remove boa in selected_action
            pre_selected_action_sample = wrap_action_sequence(tokenizer, pre_selected_action_ids)[1:].unsqueeze(0)
            pre_selected_action_mask = torch.ones_like(pre_selected_action_sample, dtype=torch.long)
            # append selected_action to sample at the end
            pre_pos_inputs.input_ids = torch.cat([pre_pos_inputs.input_ids, pre_selected_action_sample], dim=-1)
            pre_pos_inputs.attention_mask = torch.cat([pre_pos_inputs.attention_mask, pre_selected_action_mask], dim=-1)


            # process previous actions
            previous_actions = action_list[0:cur_idx]
            # action_id  fast encoding
            previous_action_ids = action_tokenizer(previous_actions)[0]
            previous_action_ids = [last_token_id - id for id in previous_action_ids]
            # 3. append action_id to sample at the begin
            # previous_action_sample = wrap_action_sequence(tokenizer, previous_action_ids).unsqueeze(0)
            previous_action_sample = torch.tensor(previous_action_ids, dtype=torch.long).unsqueeze(0)
            previous_action_mask = torch.ones_like(previous_action_sample, dtype=torch.long)

            pos_inputs.input_ids = torch.cat([pos_inputs.input_ids[:,:-1], previous_action_sample, pos_inputs.input_ids[:,-1:]], dim=-1)
            pos_inputs.attention_mask = torch.cat([pos_inputs.attention_mask, previous_action_mask], dim=-1)

            # for key, _ in pos_inputs.items():
            #     pos_inputs[key] = torch.cat([pre_pos_inputs[key], pos_inputs[key]], dim=-1)
            pos_inputs.input_ids = torch.cat([pre_pos_inputs.input_ids, pos_inputs.input_ids], dim=-1)
            pos_inputs.attention_mask = torch.cat([pre_pos_inputs.attention_mask, pos_inputs.attention_mask], dim=-1)
            pos_inputs.token_type_ids = torch.cat([pre_pos_inputs.token_type_ids, pos_inputs.token_type_ids], dim=-1)


        logits_processor = [
            ActionIDConstraintLogitsProcessor(allowed_token_ids),
        ]
        outputs = model.generate(
            pos_inputs.input_ids.to(device),
            GENERATION_CONFIG,
            max_new_tokens=50,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
        )
        outputs = outputs[:, pos_inputs.input_ids.shape[-1]:-1]
        processed_outputs = torch.tensor(last_token_id, device=outputs.device) - outputs
        processed_output_list = processed_outputs.cpu().numpy().tolist()

        action = torch.from_numpy(
            robust_decode_action(action_tokenizer, processed_output_list, action_predict_frame, 3)[0]
        ).to(device)


        action_denorm = 0.5 * (action + 1) * (action_high - action_low) + action_low

        output_dict = {
            "action": action_denorm.cpu().numpy().tolist(),
            "raw_tokens": processed_output_list,  # for GPU vs NPU comparison
        }

        if DEBUG:
            gt_action = torch.tensor(action_list[cur_idx:cur_idx + action_predict_frame]).to(device)
            gt_action_denorm = 0.5 * (gt_action + 1) * (action_high - action_low) + action_low
            output_dict["action_gt_denorm"] = gt_action_denorm.cpu().numpy().tolist()

            action_gt_id = action_tokenizer(gt_action.cpu())
            action_gt_decode = robust_decode_action(action_tokenizer, action_gt_id, action_predict_frame, 3)
            action_gt_decode = torch.tensor(action_gt_decode).to(device)[0]
            gt_action_denorm_decode = 0.5 * (action_gt_decode + 1) * (action_high - action_low) + action_low
            output_dict["action_gt_denorm_decode"] = gt_action_denorm_decode.cpu().numpy().tolist()

        output_path = os.path.join(CONFIG["output_dir"], f"{token_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)

        pbar.set_postfix_str(f"Saved {token_name}.json")
        pbar.update(1)

    pbar.close()

    dist.barrier()
    if rank == 0:
        print("All processes done.")

if __name__ == "__main__":
    main()
