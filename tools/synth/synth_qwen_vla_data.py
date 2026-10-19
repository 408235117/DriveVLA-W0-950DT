#!/usr/bin/env python3
"""Generate a tiny synthetic NavSim-style dataset for Qwen VLA throughput tests.

Produces:
  <out_root>/synth_navsim.pkl     list of scenes matching data_qwen_vla.py
  <out_root>/images/<log>/CAM_F0/<frame>.jpg

Images are 504x280, which the Qwen processor turns into exactly 10x18 = 180
vision tokens (the grid hardcoded by the ROSS forward). Actions are random in
[-1, 1]; they are valid FAST input but carry no learning signal.

Usage:
  python tools/synth/synth_qwen_vla_data.py --num_scenes 64
  # then:
  #   --data_path logs/synth_qwen_data/synth_navsim.pkl
  #   --data_root logs/synth_qwen_data/images
"""

import argparse
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw

WINDOW = 12
CUR_IDX = 3
IMG_W, IMG_H = 504, 280  # -> 18 x 10 = 180 tokens at 28 px/token


def make_image(path, scene, frame):
    base = int((scene * 37 + frame * 11) % 256)
    img = Image.new("RGB", (IMG_W, IMG_H), (base, (base * 3) % 256, (base * 7) % 256))
    draw = ImageDraw.Draw(img)
    x0 = (frame * 37) % (IMG_W - 40)
    y0 = (frame * 23) % (IMG_H - 20)
    draw.rectangle([x0, y0, x0 + 40, y0 + 20], fill=((255 - base), base, 128))
    img.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", default="logs/synth_qwen_data")
    ap.add_argument("--num_scenes", type=int, default=64)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_root = os.path.join(args.out_root, "images")
    pkl_path = os.path.join(args.out_root, "synth_navsim.pkl")
    os.makedirs(img_root, exist_ok=True)

    scenes = []
    for s in range(args.num_scenes):
        log = f"synth_{s:04d}"
        cam_dir = os.path.join(img_root, log, "CAM_F0")
        os.makedirs(cam_dir, exist_ok=True)

        image_paths = []
        for f in range(args.window):
            name = f"{f:06d}"
            make_image(os.path.join(cam_dir, name + ".jpg"), s, f)
            image_paths.append(f"/synthetic/{log}/{name}.npy")

        actions = rng.uniform(-1.0, 1.0, size=(args.window, 3)).astype(np.float32)
        prompts = ["go straight"] * args.window

        scenes.append(
            {
                "token": log,
                "text": prompts,
                "image": image_paths,
                "action": actions,
                "pre_1s_text": prompts,
                "pre_1s_image": image_paths,
                "pre_1s_action": actions.copy(),
            }
        )

    with open(pkl_path, "wb") as f:
        pickle.dump(scenes, f)

    print(f"wrote {len(scenes)} scenes -> {pkl_path}")
    print(f"images -> {img_root}")
    print(f"--data_path {pkl_path}")
    print(f"--data_root {img_root}")


if __name__ == "__main__":
    main()
