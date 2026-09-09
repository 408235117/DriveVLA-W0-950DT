#!/usr/bin/env python3
"""Fix hardcoded paths in DriveVLA-W0 pickle files (recursive, handles lists)."""

import pickle, argparse
from pathlib import Path


def _replace(obj, old, new):
    c = 0
    if isinstance(obj, str):
        if old in obj:
            return obj.replace(old, new, 1), 1
        return obj, 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            v2, c2 = _replace(v, old, new)
            if c2:
                obj[k] = v2
                c += c2
        return obj, c
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            v2, c2 = _replace(v, old, new)
            if c2:
                obj[i] = v2
                c += c2
        return obj, c
    if isinstance(obj, tuple):
        r = [_replace(v, old, new) for v in obj]
        c = sum(x[1] for x in r)
        return tuple(x[0] for x in r), c
    return obj, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path")
    parser.add_argument("--old_prefix",
                        default="/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data")
    parser.add_argument("--new_prefix", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print count and exit without saving")
    args = parser.parse_args()

    p = Path(args.pkl_path)
    with open(p, "rb") as f:
        data = pickle.load(f)

    fixed = 0
    skipped = 0
    for item in data:
        _, c = _replace(item, args.old_prefix, args.new_prefix)
        fixed += c
        if c == 0:
            skipped += 1

    print(f"\nEntries:     {len(data)}")
    print(f"Fixed:       {fixed}")
    print(f"Skipped:     {skipped}")

    if args.dry_run:
        return

    out = p.parent / f"{p.stem}_fixed.pkl"

    # Remove stale _fixed.pkl from a previous run so we start clean
    if out.exists():
        out.unlink()
        print(f"\nRemoved previous fixed copy: {out}")

    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(f"\nDone — saved to: {out}")
    print(f"\nTo use the fixed pickle for training, replace the original:")
    print(f"  mv {out} {p}")


if __name__ == "__main__":
    main()
