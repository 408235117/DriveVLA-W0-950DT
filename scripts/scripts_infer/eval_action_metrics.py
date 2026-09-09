#!/usr/bin/env python3
"""
Evaluate VLA inference results: compute ADE and FDE.

Each JSON file has:
  - action:          predicted (8, 3) denormalized actions [dx, dy, dtheta]
  - action_gt_denorm: ground truth (8, 3) denormalized actions

Usage:
  python scripts/scripts_infer/eval_action_metrics.py <json_output_dir>
"""
import json, os, sys
import numpy as np


def relative_to_absolute(actions):
    """Convert relative (dx, dy, dtheta) to absolute (x, y) positions."""
    positions = np.zeros((len(actions), 2))
    x, y, theta = 0.0, 0.0, 0.0
    for i, row in enumerate(actions):
        dx, dy = row[0], row[1]
        dtheta = row[2] if len(row) >= 3 else 0.0
        x += dx * np.cos(theta) - dy * np.sin(theta)
        y += dx * np.sin(theta) + dy * np.cos(theta)
        theta += dtheta
        positions[i] = [x, y]
    return positions


def main():
    json_dir = sys.argv[1]
    ade_list, fde_list = [], []
    errors = 0

    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)

        pred = np.array(data["action"])
        gt = np.array(data.get("action_gt_denorm", data.get("action")))

        if len(pred) == 0 or np.all(pred == 0):
            errors += 1
            continue

        pred_pos = relative_to_absolute(pred)  # (8, 3) or (8, 2)
        gt_pos = relative_to_absolute(gt)

        dists = np.linalg.norm(pred_pos - gt_pos, axis=1)  # (8,)
        ade_list.append(dists.mean())
        fde_list.append(dists[-1])

    if not ade_list:
        print("No valid samples found.")
        return

    ade = np.array(ade_list)
    fde = np.array(fde_list)
    print(f" Samples: {len(ade_list)} (zero actions skipped: {errors})")
    print(f" Min ADE:  {ade.min():.4f} m")
    print(f" Mean ADE: {ade.mean():.4f} m")
    print(f" Median ADE: {np.median(ade):.4f} m")
    print(f" Max ADE:  {ade.max():.4f} m")
    print(f" Min FDE:  {fde.min():.4f} m")
    print(f" Mean FDE: {fde.mean():.4f} m")
    print(f" Median FDE: {np.median(fde):.4f} m")
    print(f" Max FDE:  {fde.max():.4f} m")


if __name__ == "__main__":
    main()