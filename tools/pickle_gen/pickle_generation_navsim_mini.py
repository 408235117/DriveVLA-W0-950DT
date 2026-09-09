#!/usr/bin/env python3
"""
Generate a combined training pickle from NavSim mini split annotation logs.

This script reads per-scene NavSim log pickle files, extracts text/image/action
data, and saves a single combined .pkl file compatible with the Qwen VLA training
dataloader (data_qwen_vla.py).

Unlike the full pickle_generation_navsim_pre_1s.py, this script:
- Does NOT require VQ code .npy files (Qwen VLA reads raw .jpg from sensor_blobs)
- Does NOT require NuPlan data (pre_1s fallback skipped)
- Works with any NavSim split (mini, trainval, test)

Usage:
    python tools/pickle_gen/pickle_generation_navsim_mini.py \
        --logs_dir /data/lmc/navsim/download/mini_navsim_logs/mini \
        --scene_filter navtrain \
        --output_dir /data/lmc/navsim/download \
        --output_name navsim_mini_train.pkl
"""

import os
import os.path as osp
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import yaml

# project-specific imports (from the navsim devkit)
from pyquaternion import Quaternion
import sys

# Add the pickle_gen directory for navsim_coor
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
# Add repo root for project imports (e.g. utils.datasets)
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(_SCRIPT_DIR))))
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array, normalize_angle

# --- constants ---
WINDOW = 12       # frames: -3 … +8
CENTER_IDX = 3    # current frame index within window

TEXT_NAME_LIST = [
    "go left",
    "go straight",
    "go right",
    "unknown",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate NavSim training pickle for Qwen VLA")
    p.add_argument("--logs_dir", type=str, required=True,
                   help="Path to NavSim per-scene log pickle files (e.g. mini_navsim_logs/mini)")
    p.add_argument("--scene_filter", type=str, default="navtrain",
                   help="Scene filter name (navtrain, navtest, mini, etc.)")
    p.add_argument("--split", type=str, default="mini",
                   help="Split name for output naming (mini, trainval, test)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save the output pickle and norm stats")
    p.add_argument("--output_name", type=str, default=None,
                   help="Output pickle filename (default: navsim_emu_vla_256_144_{split}_pre_1s.pkl)")
    return p.parse_args()


def main():
    args = parse_args()

    logs_dir = args.logs_dir
    split = args.split
    scene_filter = args.scene_filter

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"navsim_emu_vla_256_144_{split}_pre_1s.pkl"

    # Find the scene filter YAML from the repo
    repo_root = osp.dirname(osp.dirname(osp.dirname(_SCRIPT_DIR)))
    yaml_file = osp.join(
        repo_root,
        "inference/navsim/navsim/navsim/planning/script/config/common/"
        f"train_test_split/scene_filter/{scene_filter}.yaml",
    )

    # Load scene token list
    if osp.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            scene_list = yaml.safe_load(f)
        token_list = scene_list.get('tokens', [])
        print(f"Loaded {len(token_list)} scene tokens from {yaml_file}")
    else:
        # If YAML doesn't exist, collect all scene tokens from the logs directory
        print(f"Scene filter YAML not found: {yaml_file}")
        print(f"Collecting all scene tokens from {logs_dir}...")
        token_list = []
        for fname in sorted(os.listdir(logs_dir)):
            if fname.endswith('.pkl'):
                token_list.append(fname.replace('.pkl', ''))
        print(f"Found {len(token_list)} scene tokens from log files")

    # --- Phase 1: Build scene_dict_all from per-scene logs ---
    # Keyed by scene_name (from log filename); each value is a list of per-frame dicts.
    scene_dict_all = {}

    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.pkl')]
    print(f"Processing {len(log_files)} log files from {logs_dir}")

    for log_name in tqdm(log_files, desc="Processing logs"):
        log_path = osp.join(logs_dir, log_name)
        scene = pickle.load(open(log_path, "rb"))
        num_frames = len(scene)

        # Derive scene name from log filename (e.g. "2021.05.12...pkl" -> "2021.05.12...")
        scene_name = log_name.replace('.pkl', '')

        # 1. Read all global SE2 poses
        global_ego_poses = []
        for fi in scene:
            t = fi["ego2global_translation"]
            q = Quaternion(*fi["ego2global_rotation"])
            yaw = q.yaw_pitch_roll[0]
            global_ego_poses.append([t[0], t[1], yaw])
        global_ego_poses = np.array(global_ego_poses, dtype=np.float64)

        # 2. Compute rel_all[i,j]: relative pose (dx, dy, dtheta)
        rel_all = []
        for i in range(num_frames):
            origin = StateSE2(*global_ego_poses[i])
            rel = convert_absolute_to_relative_se2_array(origin, global_ego_poses)
            rel_all.append(rel)
        rel_all = np.stack(rel_all, axis=0)

        # 3. For each frame, build action_list, image_list, text_list
        for i, fi in enumerate(scene):
            idxs = list(range(i - CENTER_IDX, i - CENTER_IDX + WINDOW))

            # 3.1 action_list: relative (dx, dy, dtheta) between consecutive frames
            action_list = []
            for j in idxs:
                if 0 <= j < num_frames - 1:
                    dx, dy, dtheta = rel_all[j, j + 1]
                else:
                    dx = dy = dtheta = 0.0
                action_list.append([float(dx), float(dy), float(dtheta)])
            fi["relative_action_list"] = action_list

            # 3.2 image_list: store camera .npy paths (resolved to .jpg by dataloader)
            # The Qwen VLA dataset code converts these paths to:
            #   <data_root>/<scene_token>/CAM_F0/<frame>.jpg
            # So we store paths that encode the scene_token and frame filename.
            image_list = []
            for j in idxs:
                if 0 <= j < num_frames:
                    cam_path = scene[j]['cams']['CAM_F0']['data_path']
                    # cam_path looks like: scene_token/CAM_F0/frame.jpg
                    # Store as-is; the dataset replaces .jpg -> .npy, then resolves
                    image_list.append(cam_path)
                else:
                    image_list.append(None)
            fi["image_vq_list"] = image_list

            # 3.3 text_list: driving command text
            text_list = []
            for j in idxs:
                if 0 <= j < num_frames:
                    driving_command = scene[j]['driving_command']
                    cmd_idx = driving_command.nonzero()[0].item()
                    text_list.append(TEXT_NAME_LIST[cmd_idx])
                else:
                    text_list.append(None)
            fi["text_list"] = text_list

            # 3.4 pre_1s data: use 2-seconds-prior frame (index i-2)
            if i < 2:
                fi["pre_1s_relative_action_list"] = fi["relative_action_list"]
                fi["pre_1s_text_list"] = fi["text_list"]
                fi["pre_1s_image_vq_list"] = fi["image_vq_list"]
            else:
                fi["pre_1s_relative_action_list"] = scene[i - 2]["relative_action_list"]
                fi["pre_1s_text_list"] = scene[i - 2]["text_list"]
                fi["pre_1s_image_vq_list"] = scene[i - 2]["image_vq_list"]

            # 3.5 Store in scene_dict_all keyed by (scene_name, frame_index)
            per_frame_token = fi.pop("token")
            _ = per_frame_token  # not used for matching — see Phase 2
            scene_dict_all.setdefault(scene_name, []).append(fi)

    # --- Phase 2: Build result_file ordered by token_list ---
    result_file = []
    missing = 0
    for scene_name in tqdm(token_list, desc="Generating result_file"):
        frame_list = scene_dict_all.get(scene_name)
        if frame_list is None:
            missing += 1
            continue
        for fi in frame_list:
            result_file.append({
                "token": scene_name,
                "text": fi["text_list"],
                "image": fi["image_vq_list"],
                "action": fi["relative_action_list"],
                "pre_1s_text": fi["pre_1s_text_list"],
                "pre_1s_image": fi["pre_1s_image_vq_list"],
                "pre_1s_action": fi["pre_1s_relative_action_list"],
            })

    print(f"Total scenes in result: {len(result_file)}")
    if missing > 0:
        print(f"Missing scenes (not in logs): {missing}")

    # --- Phase 3: Normalize actions ---
    # Inline minimal normalize_pi0 (module not available in this repo)
    from dataclasses import dataclass

    @dataclass
    class NormStats:
        mean: np.ndarray
        std: np.ndarray
        q01: np.ndarray
        q99: np.ndarray

    class RunningStats:
        """Accumulate per-element running statistics for action normalization."""
        def __init__(self):
            self.data = []

        def update(self, x):
            self.data.append(x.reshape(-1, x.shape[-1]))

        def get_statistics(self):
            all_data = np.concatenate(self.data, axis=0)
            return NormStats(
                mean=all_data.mean(axis=0),
                std=all_data.std(axis=0),
                q01=np.percentile(all_data, 1, axis=0),
                q99=np.percentile(all_data, 99, axis=0),
            )

    def save(norm_path, stats_dict):
        """Save normalization stats as a pickle file."""
        os.makedirs(norm_path, exist_ok=True)
        stat_path = osp.join(norm_path, "stat.pkl")
        with open(stat_path, "wb") as f:
            pickle.dump(stats_dict, f)
        print(f"Saved norm stats to {stat_path}")

    norm_path = osp.join(output_dir, f"normalizer_navsim_{split}")
    normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in result_file])
    normalizer.update(action_data)
    norm_stats = normalizer.get_statistics()

    print(f"Mean: {norm_stats.mean}")
    print(f"Std:  {norm_stats.std}")
    print(f"Q01:  {norm_stats.q01}")
    print(f"Q99:  {norm_stats.q99}")

    norm_stats_save = {"libero": norm_stats}
    save(norm_path, norm_stats_save)

    # Normalize + clip actions
    for scene in result_file:
        for key in ["action", "pre_1s_action"]:
            val = scene[key].copy()
            normalized = 2 * (val - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
            scene[key] = np.clip(normalized, -1, 1)

    # --- Save ---
    output_path = osp.join(output_dir, output_name)
    with open(output_path, "wb") as f:
        pickle.dump(result_file, f)

    print(f"Saved: {output_path}")
    print(f"Norm stats: {norm_path}")
    print("Done.")


if __name__ == "__main__":
    main()