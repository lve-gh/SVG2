#!/usr/bin/env python3
"""Convert PSG4D-HOI (or raw HOI4D action labels) into the forecasting annotation format.

Writes ``<out_dir>/psg4d_hoi_{train,val,test}.json`` and one per-frame COCO-RLE mask file per
video under ``<out_dir>/masks/psg4d_hoi/``. The record format is documented in
``traser_train/forecast/data_forecast.py``.

Source ``psg4d`` (the PSG4D-HOI scene graphs, data/HOI of github.com/Jingkang50/PSG4D):

    python traser/data/prepare_psg4d_hoi.py psg4d --psg4d_json HOI/psg4d_hoi.json \
        --categories HOI/categories.json --hoi4d_root HOI/HOI4D_release \
        --mask_root HOI/HOI4D_mask --out_dir traser/data

  The record layout follows the authors' visualization script (data/HOI/generate_video.py):
  ``data[i] = {"video_id": <HOI4D sequence path>, "object": [{"object_id", "mask_id"}],
  "relations": [[subject_object_id, object_object_id, predicate, start_frame, end_frame]]}``,
  class id = ``mask_id % 1000``, masks ``<mask_root>/<video_id>/mask/<frame:05d>.npz`` (key
  ``array``, stored transposed). Check these assumptions on the first files you convert.

Source ``hoi4d`` (fallback without PSG4D: HOI4D's own action segmentation + 2D masks; this is
how PSG4D pre-annotated its relations — "hand <action> <object>"):

    python traser/data/prepare_psg4d_hoi.py hoi4d --hoi4d_root HOI4D_release \
        --annotation_root HOI4D_annotations --instructions HOI4D-Instructions --out_dir traser/data

Splits are by object instance (``C*/N*`` of the HOI4D path): test objects are never seen in
training, which keeps clip-level leakage out of the benchmark.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as maskUtils

FPS = 15.0
IDLE_EVENTS = {"rest", "stop"}


def rle(binary: np.ndarray) -> dict:
    r = maskUtils.encode(np.asfortranarray(binary.astype(np.uint8)))
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": r["counts"].decode("ascii")}


def split_of(seq: str, val_pct: int, test_pct: int) -> str:
    parts = seq.strip("/").split("/")
    instance = "/".join(p for p in parts if p[:1] in ("C", "N"))  # e.g. "C5/N50"
    h = int(hashlib.md5(instance.encode()).hexdigest(), 16) % 100
    return "test" if h < test_pct else ("val" if h < test_pct + val_pct else "train")


def merge_spans(rels):
    """[[s, p, o, start, end], ...] -> [[s, p, o, [[start, end], ...]], ...]"""
    spans = defaultdict(list)
    for s, o, p, a, b in rels:
        spans[(s, p, o)].append([int(a), int(b)])
    return [[s, p, o, sorted(v)] for (s, p, o), v in spans.items()]


# --------------------------------------------------------------------------- PSG4D-HOI
def convert_psg4d(args):
    data = json.load(open(args.psg4d_json))
    data = data["data"] if isinstance(data, dict) else data
    categories = json.load(open(args.categories))
    mask_dir = os.path.join(args.out_dir, "masks", "psg4d_hoi")
    os.makedirs(mask_dir, exist_ok=True)
    out = defaultdict(list)
    for k, d in enumerate(data):
        seq = d["video_id"]
        video = os.path.join(args.hoi4d_root, seq, "align_rgb", "image.mp4")
        frames = sorted(glob.glob(os.path.join(args.mask_root, seq, "mask", "*.npz")))
        if not os.path.exists(video) or not frames:
            print(f"skip {seq}: video or masks missing", file=sys.stderr)
            continue
        objs = d["object"]
        col = {o["object_id"]: j for j, o in enumerate(objs)}
        objects = [{"id": j, "name": categories[str(o["mask_id"] % 1000)]} for j, o in enumerate(objs)]
        stem = seq.replace("/", "_")
        mask_json = os.path.join(mask_dir, f"{stem}_rle.json")
        if not os.path.exists(mask_json):
            per_frame = []
            for f in frames:
                m = np.load(f)["array"].T
                per_frame.append([rle(m == o["mask_id"]) for o in objs])
            json.dump(per_frame, open(mask_json, "w"))
        rels = [[col[r[0]], col[r[1]], r[2], r[3], r[4]] for r in d["relations"]
                if r[0] in col and r[1] in col]
        out[split_of(seq, args.val_pct, args.test_pct)].append(dict(
            video=os.path.abspath(video), mask_json=os.path.abspath(mask_json), fps=FPS,
            num_frames=len(frames), objects=objects, relationships=merge_spans(rels), source=seq))
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(data)}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- HOI4D actions
def color_map(n=256):
    """HOI4D's label -> RGB palette (utils/color_map.py of HOI4D-Instructions)."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


def convert_hoi4d(args):
    import csv
    import cv2

    labels = {}
    with open(os.path.join(args.instructions, "definitions", "motion segmentation", "label.csv")) as f:
        for row in csv.reader(f):
            if row and row[0].startswith("C"):
                labels[row[0]] = [x for x in row[2:] if x]
    cmap = color_map()
    seqs = [l.strip() for l in open(os.path.join(args.instructions, "release.txt")) if l.strip()]
    mask_dir = os.path.join(args.out_dir, "masks", "psg4d_hoi")
    os.makedirs(mask_dir, exist_ok=True)
    out = defaultdict(list)
    for seq in seqs:
        video = os.path.join(args.hoi4d_root, seq, "align_rgb", "image.mp4")
        ann = os.path.join(args.annotation_root, seq)
        action = os.path.join(ann, "action", "color.json")
        pngs = sorted(glob.glob(os.path.join(ann, "2Dseg", "*", "*.png")))
        if not (os.path.exists(video) and os.path.exists(action) and pngs):
            continue
        cat = [p for p in seq.split("/") if p.startswith("C")][0]
        names = labels[cat]                      # label k+1 -> names[k]
        hand = names.index("Right Hand") + 1
        # Read every frame's colour-coded mask once; keep the labels that ever appear.
        frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in pngs]
        present = []
        for lab in range(1, len(names) + 1):
            col = cmap[lab]
            if any((fr == col).all(-1).any() for fr in frames[:: max(1, len(frames) // 20)]):
                present.append(lab)
        objects = [{"id": j, "name": names[lab - 1].lower()} for j, lab in enumerate(present)]
        stem = seq.replace("/", "_")
        mask_json = os.path.join(mask_dir, f"{stem}_rle.json")
        if not os.path.exists(mask_json):
            json.dump([[rle((fr == cmap[lab]).all(-1)) for lab in present] for fr in frames], open(mask_json, "w"))
        # Action segments: "hand <event> <main object>" (label 1 is the task's object).
        # {"events": [{"event": "Reachout", "startTime": 1.359, "endTime": 2.901}, ...]}; "rest" and
        # "Stop" are idle phases, not interactions (PSG4D-HOI has no such predicates).
        a = json.load(open(action, encoding="utf-8"))
        rels = []
        if hand in present and 1 in present:
            s, o = present.index(hand), present.index(1)
            for e in a.get("events", []):
                name = e.get("event", "").strip()
                t0, t1 = float(e["startTime"]), float(e["endTime"])
                if name and name.lower() not in IDLE_EVENTS and t1 > t0:
                    rels.append([s, o, name.lower(), int(t0 * FPS), min(int(t1 * FPS), len(frames) - 1)])
        out[split_of(seq, args.val_pct, args.test_pct)].append(dict(
            video=os.path.abspath(video), mask_json=os.path.abspath(mask_json), fps=FPS, num_frames=len(frames),
            objects=objects, relationships=merge_spans(rels), source=seq))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    p = sub.add_parser("psg4d")
    p.add_argument("--psg4d_json", required=True)
    p.add_argument("--categories", required=True)
    p.add_argument("--hoi4d_root", required=True)
    p.add_argument("--mask_root", required=True)
    h = sub.add_parser("hoi4d")
    h.add_argument("--hoi4d_root", required=True)
    h.add_argument("--annotation_root", required=True)
    h.add_argument("--instructions", required=True, help="Clone of github.com/leolyliu/HOI4D-Instructions")
    for x in (p, h):
        x.add_argument("--out_dir", default="traser/data")
        x.add_argument("--val_pct", type=int, default=10)
        x.add_argument("--test_pct", type=int, default=15)
    args = ap.parse_args()

    out = convert_psg4d(args) if args.source == "psg4d" else convert_hoi4d(args)
    for split, items in out.items():
        path = os.path.join(args.out_dir, f"psg4d_hoi_{split}.json")
        json.dump(items, open(path, "w"), indent=1)
        n_rel = sum(len(r[3]) for it in items for r in it["relationships"])
        print(f"{split}: {len(items)} videos, {n_rel} relation spans -> {path}")


if __name__ == "__main__":
    main()
