#!/usr/bin/env python3
"""Run TRASER on a video + its object mask trajectories and print the scene graph.

The preprocessing mirrors training (``traser_train/data/data_qwen.py``) exactly — the
same frame sampling, the same video processor limits, the same mask -> token selection,
and the same trajectory-aligned token arrangement — so inference sees sequences of the
same shape the model was trained on.

    python traser/inference.py --video clip.mp4 --masks clip_rle.json

Weights come from the Hub (``UWGZQ/TRASER``) unless ``--model`` points at a local
directory. ``--masks`` is the per-frame, per-object COCO RLE JSON produced by
``traser/data/prepare_svg2.py`` or by the SVG2 annotation pipeline
(``pipeline/svg2_pipeline.py``, whose ``stage6_scene_graph.json`` can be converted with
``--from_scene_graph``). Progress goes to stderr; stdout is the scene graph only.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens

DEFAULT_MODEL = "UWGZQ/TRASER"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Qwen2.5-VL vision constants (traser_train/train/trainer_insert.py).
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
OBJECT_LABEL_TEMPLATE = "Object {i}: "

# Video sampling, identical to DataArguments' defaults used for training.
BASE_INTERVAL = 1           # seconds between sampled frames
VIDEO_MIN_FRAMES = 4
VIDEO_MAX_FRAMES = 128
VIDEO_MAX_FRAME_PIXELS = 768 * 28 * 28
VIDEO_MIN_FRAME_PIXELS = 64 * 28 * 28

SYSTEM_MESSAGE = "You are a helpful assistant."
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

PROMPTS = {
    "scene_graph": "Output the Video Scene Graph from the video and object trajectories:\n<video>\n",
    "relationships": "List all objects and their relationships from the video and object trajectories:\n<video>\n",
    "attributes": "List all objects and their attributes from the video and object trajectories:\n<video>\n",
    "objects": "List all objects from the video and object trajectories:\n<video>\n",
}


# --------------------------------------------------------------------------- video
def decode_video(path, video_processor):
    """Sample ~1 frame/second and preprocess, exactly as the training dataset does."""
    try:
        from decord import VideoReader
        vr = VideoReader(path, num_threads=4)
        total_frames, avg_fps = len(vr), vr.get_avg_fps()
        frame_idx = _frame_indices(total_frames, total_frames / avg_fps)
        video = vr.get_batch(frame_idx).asnumpy()
    except Exception as exc:  # noqa: BLE001 - fall back like the training loader
        print(f"[traser] decord failed ({exc}); falling back to torchcodec", file=sys.stderr)
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(path, device="cpu")
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        frame_idx = _frame_indices(total_frames, total_frames / avg_fps)
        video = decoder.get_frames_at(indices=frame_idx.tolist()).data.cpu().numpy()

    video_length = total_frames / avg_fps
    fps = len(frame_idx) / video_length

    import copy as _copy
    proc = _copy.deepcopy(video_processor)
    proc.max_pixels = VIDEO_MAX_FRAME_PIXELS
    proc.min_pixels = VIDEO_MIN_FRAME_PIXELS
    proc.size["longest_edge"] = proc.max_pixels
    proc.size["shortest_edge"] = proc.min_pixels
    out = proc.preprocess(videos=video, return_tensors="pt")
    second_per_grid_ts = [video_processor.temporal_patch_size / fps] * len(out["video_grid_thw"])
    return (out["pixel_values_videos"], out["video_grid_thw"][0],
            frame_idx.tolist(), second_per_grid_ts)


def _frame_indices(total_frames, video_length):
    target = min(max(round(video_length / BASE_INTERVAL), VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES)
    return np.unique(np.linspace(0, total_frames - 1, target, dtype=int))


# --------------------------------------------------------------------------- masks
def build_obj_masks(mask_data, obj_ids, sampled_idx, h_rz, w_rz):
    """(O, N, h_rz, w_rz) binary masks; also returns the ids whose masks are non-empty."""
    masks = torch.zeros((len(obj_ids), len(sampled_idx), h_rz, w_rz), dtype=torch.float32)
    for o_idx, oid in enumerate(obj_ids):
        for n_idx, f_idx in enumerate(sampled_idx):
            if not (0 <= f_idx < len(mask_data)):
                continue
            frame = mask_data[f_idx]
            if not frame or not (0 <= oid < len(frame)):
                continue
            rle = frame[oid]
            if not rle:
                continue
            # Absent objects are stored as all-zero RLEs; skip them without decoding. Some
            # released placeholders encode a run shorter than size[0] * size[1], and
            # pycocotools then leaves the rest of its output buffer uninitialised, so
            # decoding them returns garbage instead of zeros. area() only reads run lengths.
            if maskUtils.area({"size": rle["size"], "counts": rle["counts"]}) == 0:
                continue
            m = maskUtils.decode({"size": rle["size"], "counts": rle["counts"]})
            if m.ndim == 3:
                m = m[:, :, 0]
            m_t = torch.from_numpy(m.astype(np.uint8))[None, None].float()
            masks[o_idx, n_idx] = (F.interpolate(m_t, size=(h_rz, w_rz), mode="nearest")[0, 0] > 0.5).float()

    keep = (masks.view(len(obj_ids), -1).sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(1).tolist()
    if not keep:
        raise SystemExit("None of the requested objects has a mask on the sampled frames.")
    return masks[keep], [obj_ids[i] for i in keep]


def masks_from_scene_graph(path):
    """Convert a pipeline ``stage6_scene_graph.json`` into the per-frame RLE mask list."""
    sg = json.load(open(path))
    objects = sg["objects"]
    n_frames = sg["total_frames"]
    empty = {"size": [sg["height"], sg["width"]], "counts": maskUtils.encode(
        np.asfortranarray(np.zeros((sg["height"], sg["width"]), dtype=np.uint8)))["counts"].decode()}
    out = [[dict(empty) for _ in objects] for _ in range(n_frames)]
    for o_idx, obj in enumerate(objects):
        # trajectory["masks"] is already dense and self-indexed: position i IS video frame
        # i, null/falsy where the object is absent. trajectory["frames"] is only the sparse
        # list of frame indices where masks[i] is non-empty (derived from it in the
        # pipeline's stage5_structure) -- it must not be zipped against masks.
        for f_idx, rle in enumerate(obj["trajectory"]["masks"]):
            if rle and 0 <= f_idx < n_frames:
                out[f_idx][o_idx] = {"size": rle["size"], "counts": rle["counts"]}
    return out


# --------------------------------------------------------------------------- prompt
def build_prompt_ids(tokenizer, prompt, n_video_tokens):
    tok = tokenizer
    saved = getattr(tok, "chat_template", None)
    tok.chat_template = CHAT_TEMPLATE
    try:
        content = prompt.replace(
            "<video>", "<|vision_start|>" + "<|video_pad|>" * n_video_tokens + "<|vision_end|>"
        )
        ids = list(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_MESSAGE}], return_dict=False))
        ids += list(tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, return_dict=False))
    finally:
        tok.chat_template = saved
    return torch.tensor([ids], dtype=torch.long)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Input video file.")
    ap.add_argument("--masks", required=True,
                    help="Per-frame, per-object COCO RLE JSON, or a pipeline "
                         "stage6_scene_graph.json with --from_scene_graph.")
    ap.add_argument("--from_scene_graph", action="store_true",
                    help="Read --masks as a pipeline stage-6 scene graph instead of an RLE list.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hub id or local checkpoint directory.")
    ap.add_argument("--task", default="scene_graph", choices=sorted(PROMPTS),
                    help="Which of the four training prompts to use.")
    ap.add_argument("--objects", type=int, nargs="*", default=None,
                    help="Object ids (columns of the mask JSON) to describe. Default: all.")
    ap.add_argument("--max_objects", type=int, default=40,
                    help="Cap on the number of objects, matching the released training data.")
    ap.add_argument("--coverage_thresh", type=float, default=0.5)
    ap.add_argument("--time_reduce", default="max", choices=["mean", "max", "min"])
    ap.add_argument("--temporal_window_length", type=int, default=4, help="Seconds per window.")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--output", default=None, help="Write the generated scene graph here.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TRASER.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    processor.tokenizer = tokenizer

    # ---- video ----
    pixel_values_videos, video_grid_thw, sampled_idx, second_per_grid_ts = decode_video(
        args.video, processor.video_processor)
    t_grid, h_patch, w_patch = (int(x) for x in video_grid_thw)
    n_video_tokens = t_grid * h_patch * w_patch // SPATIAL_MERGE_SIZE ** 2
    print(f"[traser] {len(sampled_idx)} frames -> grid {t_grid}x{h_patch}x{w_patch} "
          f"({n_video_tokens} video tokens)", file=sys.stderr)

    # ---- masks ----
    mask_data = masks_from_scene_graph(args.masks) if args.from_scene_graph else json.load(open(args.masks))
    obj_ids = args.objects if args.objects is not None else list(range(len(mask_data[0])))
    obj_ids = sorted(obj_ids)[: args.max_objects]
    obj_masks, obj_ids = build_obj_masks(
        mask_data, obj_ids, sampled_idx, h_patch * PATCH_SIZE, w_patch * PATCH_SIZE)
    print(f"[traser] {len(obj_ids)} objects with masks: {obj_ids}", file=sys.stderr)

    # ---- token selection + trajectory-aligned arrangement ----
    _, per_obj_idx, _ = select_tokens(
        obj_masks=obj_masks, grid_thw=(t_grid, h_patch, w_patch), patch_size=PATCH_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE, temporal_patch_size=TEMPORAL_PATCH_SIZE,
        coverage_thresh=args.coverage_thresh, time_reduce=args.time_reduce, device="cpu")

    input_ids = build_prompt_ids(tokenizer, PROMPTS[args.task], n_video_tokens).to(device)
    attention_mask = torch.ones_like(input_ids)

    label_ids = tokenizer([OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per_obj_idx))],
                          add_special_tokens=False)["input_ids"]
    seconds_per_grid = TEMPORAL_PATCH_SIZE / BASE_INTERVAL
    grids_per_window = int(args.temporal_window_length / seconds_per_grid)
    windows = [f"<{w * args.temporal_window_length} - "
               f"{(w + 1) * args.temporal_window_length} sec>"
               for w in range(math.ceil(t_grid / grids_per_window))]
    window_ids = tokenizer(windows, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        embeds, position_ids, mask, rope_deltas, _, _, _ = rearrange_token(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos.to(device, dtype=torch.bfloat16),
            video_grid_thw=video_grid_thw[None].to(device),
            image_grid_thw=None, pixel_values=None,
            second_per_grid_ts=torch.tensor(second_per_grid_ts),
            obj_token_indices_per_sample=[[i.to(device) for i in per_obj_idx]],
            obj_traj_start_id=int(model.config.obj_traj_start_id),
            obj_traj_end_id=int(model.config.obj_traj_end_id),
            text_token_ids_per_sample=[[torch.tensor(x, dtype=torch.long) for x in label_ids]],
            timestamp_token_ids_per_batch=[[torch.tensor(x, dtype=torch.long) for x in window_ids]],
            grids_per_temporal_window_per_batch=[grids_per_window],
        )
        generated = model.generate(
            inputs_embeds=embeds, position_ids=position_ids, attention_mask=mask.long(),
            rope_deltas=rope_deltas, max_new_tokens=args.max_new_tokens, do_sample=False)

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[traser] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
