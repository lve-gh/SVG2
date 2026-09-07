#!/usr/bin/env python3
"""Convert the released SVG2 dataset into the annotation format used for training.

SVG2 (https://huggingface.co/datasets/UWGZQ/Synthetic_Visual_Genome2) ships the scene
graphs and segmentation masks of all TRASER training datasets as parquet shards:

  source   HF subdirectory                 videos to download (see docs/DATA.md)
  ------   -----------------------------   ---------------------------------------
  sav      data|masks/cleaned/sav          SA-V (Meta)
  pvd      data|masks/cleaned/pvd          PE-Video (facebook/PE-Video)
  vipseg   data|masks/academic_datasets/…  VIPSeg (re-encoded at 6 fps)
  vidor    data|masks/academic_datasets/…  VidOR
  vidvrd   data|masks/academic_datasets/…  ImageNet-VidVRD
  lvvis    data|masks/academic_datasets/…  LV-VIS (re-encoded at 5 fps)
  ovis     data|masks/academic_datasets/…  OVIS (re-encoded at 5 fps)

This script reads a local snapshot of the SVG2 parquet shards plus your local copy of
the videos, and writes:

  <out_dir>/<annotation>.json                    annotation list (training format)
  <out_dir>/masks/<source>/<video_id>_rle.json   per-video, per-frame RLE masks

Example:
    python traser/data/prepare_svg2.py \
        --source lvvis \
        --svg2_root /path/to/Synthetic_Visual_Genome2 \
        --video_root /path/to/LV-VIS/videos_fps5 \
        --out_dir traser/data

The script is resumable: existing per-video mask JSONs are not rewritten.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROMPT_SCENE_GRAPH = "Output the Video Scene Graph from the video and object trajectories:\n<video>\n"
PROMPT_OBJECTS_RELATIONSHIPS = "List all objects and their relationships from the video and object trajectories:\n<video>\n"
PROMPT_OBJECTS_ATTRIBUTES = "List all objects and their attributes from the video and object trajectories:\n<video>\n"
PROMPT_OBJECTS = "List all objects from the video and object trajectories:\n<video>\n"

# Object id representing the camera/observer in relationships. Camera edges carry no
# object entry or mask; they are kept as long as the other endpoint is a kept object.
CAMERA_ID = -1

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}

SOURCES = {
    # source: (data subdir, masks subdir, output annotation filename)
    "sav": ("data/cleaned/sav", "masks/cleaned/sav", "svg2_sav.json"),
    "pvd": ("data/cleaned/pvd", "masks/cleaned/pvd", "svg2_pvd.json"),
    "vipseg": ("data/academic_datasets/vipseg", "masks/academic_datasets/vipseg", "vipseg.json"),
    "vidor": ("data/academic_datasets/vidor", "masks/academic_datasets/vidor", "vidor.json"),
    "vidvrd": ("data/academic_datasets/vidvrd", "masks/academic_datasets/vidvrd", "vidvrd.json"),
    "lvvis": ("data/academic_datasets/lvvis", "masks/academic_datasets/lvvis", "lvvis.json"),
    "ovis": ("data/academic_datasets/ovis", "masks/academic_datasets/ovis", "ovis.json"),
}


def compact_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def object_id(entry: dict) -> int:
    """Extract N from the single 'object_N' key of an objects-list entry."""
    key = next(k for k in entry.keys() if k.startswith("object_"))
    return int(key.split("_")[1])


def index_videos(video_root: str) -> dict:
    """Map video file stem (= SVG2 video_id) -> absolute path (searched recursively)."""
    index = {}
    for root, _, files in os.walk(video_root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in VIDEO_EXTENSIONS:
                index[stem] = os.path.abspath(os.path.join(root, name))
    return index


def convert_row(objects_json: str, relationships_json: str, source: str, video_id: str,
                max_object=None):
    """Turn one SVG2 data row into (obj_list, human prompt, gpt payload string,
    has_relationships).

    - object ids are taken from the released ``object_N`` keys. They may be
      non-contiguous; they are preserved verbatim so that they keep indexing the
      corresponding entries of the per-frame mask list.
    - relationships ``[subj, pred, obj, windows, rel_type]`` are reduced to the
      training format ``[subj, pred, obj, windows]``. Edges whose endpoints are not
      kept objects are dropped. Camera/observer edges (id -1) are retained whenever
      their other endpoint is a kept object.
    - ``max_object`` caps oversized samples by keeping the first ``max_object``
      objects (in stored id order), mirroring the original dataset build; the
      released TRASER model was trained with ``max_object=40`` on the pvd split.
    """
    objects = json.loads(objects_json)
    relationships = json.loads(relationships_json) if relationships_json else []

    if max_object is not None and 0 < max_object < len(objects):
        objects = objects[:max_object]

    obj_list = [object_id(o) for o in objects]
    valid_ids = set(obj_list)

    rel_payload = []
    for rel in relationships:
        subj, pred, obj, windows = rel[0], rel[1], rel[2], rel[3]
        if (subj != CAMERA_ID and subj not in valid_ids) or (obj != CAMERA_ID and obj not in valid_ids):
            continue
        if subj == CAMERA_ID and obj == CAMERA_ID:
            continue
        rel_payload.append([subj, pred, obj, windows])

    # The sav/pvd training data merges duplicate (subj, pred, obj) edges (the release
    # keeps edges of different relation types separate); the academic datasets do not.
    if source in ("sav", "pvd"):
        merged = {}
        for subj, pred, obj, windows in rel_payload:
            key = (subj, pred, obj)
            if key not in merged:
                merged[key] = [subj, pred, obj, list(windows)]
            else:
                merged[key][3] = merged[key][3] + list(windows)
        rel_payload = list(merged.values())

    has_relationships = len(rel_payload) > 0
    all_attrs_empty = not any(o.get("attributes") for o in objects)

    # The training data carries an explicit empty attribute list for attribute-less
    # objects whenever the sample has attributes at all; the release omits the key.
    if not all_attrs_empty:
        for o in objects:
            if "attributes" not in o:
                o["attributes"] = []

    if has_relationships:
        prompt = PROMPT_OBJECTS_RELATIONSHIPS if all_attrs_empty else PROMPT_SCENE_GRAPH
    else:
        prompt = PROMPT_OBJECTS if all_attrs_empty else PROMPT_OBJECTS_ATTRIBUTES

    gpt_dict = {"objects": objects}
    if has_relationships:
        gpt_dict["relationships"] = rel_payload

    return obj_list, prompt, compact_dumps(gpt_dict), has_relationships


def convert_masks(mask_dir: Path, out_mask_dir: Path) -> dict:
    """Unpack the mask parquet shards into one RLE JSON per video.

    The parquet stores a flat, frame-major list of COCO RLE count strings
    (``counts[frame_idx * num_objects + obj_idx]``); the training format is
    ``list(frames)`` of ``list(objects)`` of ``{"size": [H, W], "counts": str}``.

    Returns: video_id -> (mask_json_path, num_objects)
    """
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    mask_info = {}

    shards = sorted(mask_dir.glob("*.parquet"))
    if not shards:
        sys.exit(f"No mask parquet shards found under {mask_dir}")

    for shard_idx, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=16):
            for row_idx in range(batch.num_rows):
                video_id = batch["video_id"][row_idx].as_py()
                num_objects = int(batch["num_objects"][row_idx].as_py())
                out_path = out_mask_dir / f"{video_id}_rle.json"
                mask_info[video_id] = (str(out_path.resolve()), num_objects)
                if out_path.exists():
                    continue

                height = int(batch["mask_height"][row_idx].as_py())
                width = int(batch["mask_width"][row_idx].as_py())
                num_frames = int(batch["num_frames"][row_idx].as_py())
                counts = batch["counts"][row_idx].as_py()
                assert len(counts) == num_frames * num_objects, video_id

                frames = [
                    [
                        {"size": [height, width], "counts": counts[f * num_objects + o]}
                        for o in range(num_objects)
                    ]
                    for f in range(num_frames)
                ]
                with open(out_path, "w") as f:
                    json.dump(frames, f)
        print(f"[masks] shard {shard_idx + 1}/{len(shards)} done ({len(mask_info)} videos)")

    return mask_info


def convert_annotations(data_dir: Path, mask_info: dict, video_index: dict,
                        source: str, max_object=None):
    """Build the training annotation list from the scene-graph parquet shards."""
    annotations = []
    skipped_no_video = 0
    skipped_mismatch = 0

    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        sys.exit(f"No data parquet shards found under {data_dir}")

    for shard in shards:
        table = pq.read_table(shard)
        for i in range(table.num_rows):
            video_id = table["video_id"][i].as_py()

            video_path = video_index.get(video_id)
            if video_path is None:
                skipped_no_video += 1
                continue
            if video_id not in mask_info:
                skipped_mismatch += 1
                continue
            mask_path, num_objects = mask_info[video_id]

            obj_list, prompt, gpt_value, has_relationships = convert_row(
                table["objects"][i].as_py(),
                table["relationships"][i].as_py(),
                source,
                video_id,
                max_object=max_object,
            )
            # Object ids index into each frame's mask list, so every id must exist there.
            if not obj_list or max(obj_list) >= num_objects:
                skipped_mismatch += 1
                continue

            annotations.append(
                {
                    "video": video_path,
                    "mask_json": mask_path,
                    "obj_list": obj_list,
                    "has_relationships": has_relationships,
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": gpt_value},
                    ],
                }
            )

    return annotations, skipped_no_video, skipped_mismatch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True,
                        help="Which SVG2 source split to convert.")
    parser.add_argument("--svg2_root", required=True,
                        help="Local snapshot of UWGZQ/Synthetic_Visual_Genome2 (contains data/ and masks/).")
    parser.add_argument("--video_root", required=True,
                        help="Directory containing the downloaded videos (searched recursively; "
                             "files are matched to SVG2 video_ids by file name).")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for the annotation JSON and the masks/ subdirectory.")
    parser.add_argument("--max_object", type=int, default=None,
                        help="If set, keep at most this many objects per sample (the first N in "
                             "stored order); relationships touching dropped objects are removed. "
                             "Use --max_object 40 for the pvd split to reproduce the released "
                             "TRASER training data.")
    args = parser.parse_args()

    data_subdir, mask_subdir, out_name = SOURCES[args.source]
    svg2_root = Path(args.svg2_root)
    data_dir = svg2_root / data_subdir
    mask_dir = svg2_root / mask_subdir
    out_dir = Path(args.out_dir)

    print(f"Indexing videos under {args.video_root} ...")
    video_index = index_videos(args.video_root)
    print(f"Found {len(video_index)} video files.")

    print(f"Converting masks from {mask_dir} ...")
    mask_info = convert_masks(mask_dir, out_dir / "masks" / args.source)

    print(f"Converting annotations from {data_dir} ...")
    annotations, skipped_no_video, skipped_mismatch = convert_annotations(
        data_dir, mask_info, video_index, args.source, max_object=args.max_object
    )

    out_path = out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(compact_dumps(annotations))

    print(f"Wrote {len(annotations)} samples to {out_path}")
    if skipped_no_video:
        print(f"Skipped {skipped_no_video} samples with no matching video under --video_root.")
    if skipped_mismatch:
        print(f"Skipped {skipped_mismatch} samples with missing or mismatched masks.")


if __name__ == "__main__":
    main()
