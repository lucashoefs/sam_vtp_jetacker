#!/usr/bin/env python3
"""Strip a training checkpoint to what the robot needs, and record its hash.

A training checkpoint carries the optimiser state, which is two thirds of the
file and is never read at inference. The hash matters more than the size: an
A/B campaign whose arms cannot be told apart afterwards is not a paired
experiment, it is a pile of recordings. Every export writes a sha256 beside the
weights, and the node logs the same prefix at startup, so a bag can be traced to
the exact file that produced it.

Exports are verified before they are written out -- the file is loaded back
through `build_model` and its components printed. A checkpoint that only fails
at model-load time fails on the robot, in the field, at the start of a session.

    ./export_checkpoint.py runs/sam_tp_v5_nogeo/best_model.pt ~/deploy/sam_tp.pt
    ./export_checkpoint.py runs/sam_tp_v5_nogeo/best_model.pt ~/deploy/sam_tp_fp16.pt --fp16
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

# The model repo is wiped and re-cloned on every update, so find it rather than
# assuming an install.
for _p in (os.path.expanduser("~/sam_vtp/src"), os.path.expanduser("~/sam_vtp")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from sam_vtp import build_model, describe  # noqa: E402

KEEP = ("config", "epoch", "best_val_iou", "val_iou", "global_step")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="training checkpoint")
    ap.add_argument("dst", help="output path (a .sha256 is written beside it)")
    ap.add_argument("--fp16", action="store_true",
                    help="cast floating-point weights to half precision")
    ap.add_argument("--force", action="store_true", help="overwrite an existing export")
    a = ap.parse_args()

    dst = os.path.expanduser(a.dst)
    if os.path.exists(dst) and not a.force:
        print(f"refusing to overwrite {dst} (pass --force)")
        return 1

    raw = torch.load(os.path.expanduser(a.src), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("model", raw))
    if a.fp16:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}

    out = {"model_state_dict": state}
    out.update({k: raw[k] for k in KEEP if isinstance(raw, dict) and k in raw})

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    torch.save(out, tmp)

    # Verify before publishing the name the robot will load.
    try:
        model = build_model(tmp, device=torch.device("cpu"))
    except Exception as exc:
        os.remove(tmp)
        print(f"export FAILED to load back: {exc}")
        return 1
    os.replace(tmp, dst)

    digest = sha256(dst)
    with open(dst + ".sha256", "w") as fh:
        fh.write(f"{digest}  {os.path.basename(dst)}\n")

    src_mb = os.path.getsize(os.path.expanduser(a.src)) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print(f"{a.src}  ->  {dst}")
    print(f"  {src_mb:.0f} MB -> {dst_mb:.0f} MB ({len(state)} tensors)")
    print(f"  {describe(model)}")
    print(f"  trained: {model.trained_components}")
    print(f"  sha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
