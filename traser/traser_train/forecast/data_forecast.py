"""Forecasting dataset: clips of H observed + F future seconds cut from annotated videos.

Annotation format (one JSON list per split, written by ``traser/data/prepare_psg4d_hoi.py``):

    {
      "video":         "/abs/path/clip.mp4",
      "mask_json":     "/abs/path/clip_rle.json",   # list(frames) of list(objects) COCO RLE, every native frame
      "fps":           15.0,                        # native frame rate of the video and the masks
      "num_frames":    300,
      "objects":       [{"id": 0, "name": "hand"}, {"id": 1, "name": "bottle"}],   # id = column in mask_json
      "relationships": [[0, "grasp", 1, [[85, 89]]]]                                 # native-frame intervals, inclusive
    }

A training sample is a clip ``[T - H, T + F)`` (whole seconds). Frames are sampled at
``sample_fps`` (default 2, so that one Qwen2.5-VL temporal grid = 1 s). The model sees
only the first ``H`` seconds; the last ``F`` seconds provide the targets:

* the future relationships, as text: ``{"relationships": [[s, pred, o, [[t0, t1]]], ...]}``
  with ``t0, t1`` whole seconds counted from the clip start (``H <= t <= H + F - 1``);
* the future boxes of every object, one per forecast window (normalized cx, cy, w, h);
* the future masks, used only to compute the latent-distillation targets.
"""

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = -100
SYSTEM_MESSAGE = "You are a helpful assistant."
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
# The video is replaced by trajectory-aligned object blocks, so the prompt only needs one
# placeholder token to mark where they go.
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
FORECAST_PROMPT = (
    "Forecast the relationships between the objects in the next {F} seconds "
    "(from {H} to {E} sec), given the video and object trajectories of the first {H} seconds:\n<video>\n"
)


# --------------------------------------------------------------------------- io helpers
def load_annotations(path: str) -> List[dict]:
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def read_frames(video_path: str, frame_idx: Sequence[int]) -> np.ndarray:
    """Return frames ``frame_idx`` as uint8 RGB (N, H, W, 3); decord, else OpenCV."""
    try:
        from decord import VideoReader
        vr = VideoReader(video_path, num_threads=4)
        idx = [min(max(int(i), 0), len(vr) - 1) for i in frame_idx]
        return vr.get_batch(idx).asnumpy()
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(int(i), 0), n - 1))
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of {video_path}")
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _rle(entry):
    if not entry:
        return None
    return {"size": entry["size"], "counts": entry["counts"]}


def window_boxes(mask_data, obj_ids: Sequence[int], window_frames: Sequence[Sequence[int]]):
    """Union box of each object over the frames of each window, from the RLEs (no decoding).

    Returns ``boxes`` (O, W, 4) as normalized (cx, cy, w, h) and ``valid`` (O, W) bool.
    """
    from pycocotools import mask as maskUtils

    O, W = len(obj_ids), len(window_frames)
    boxes = torch.zeros(O, W, 4)
    valid = torch.zeros(O, W, dtype=torch.bool)
    for o, oid in enumerate(obj_ids):
        for w, frames in enumerate(window_frames):
            x0 = y0 = float("inf")
            x1 = y1 = float("-inf")
            img_h = img_w = None
            for fidx in frames:
                if not (0 <= fidx < len(mask_data)) or not (0 <= oid < len(mask_data[fidx])):
                    continue
                rle = _rle(mask_data[fidx][oid])
                if rle is None or maskUtils.area(rle) == 0:
                    continue
                bx, by, bw, bh = maskUtils.toBbox(rle).tolist()
                img_h, img_w = rle["size"]
                x0, y0 = min(x0, bx), min(y0, by)
                x1, y1 = max(x1, bx + bw), max(y1, by + bh)
            if img_h is not None:
                boxes[o, w] = torch.tensor([(x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
                                            (x1 - x0) / img_w, (y1 - y0) / img_h])
                valid[o, w] = True
    return boxes, valid


def decode_masks(mask_data, obj_ids: Sequence[int], frame_idx: Sequence[int], h_rz: int, w_rz: int) -> torch.Tensor:
    """(O, N, h_rz, w_rz) uint8 masks on the resized frame grid."""
    from pycocotools import mask as maskUtils

    out = torch.zeros((len(obj_ids), len(frame_idx), h_rz, w_rz), dtype=torch.uint8)
    for o, oid in enumerate(obj_ids):
        for n, fidx in enumerate(frame_idx):
            if not (0 <= fidx < len(mask_data)) or not (0 <= oid < len(mask_data[fidx])):
                continue
            rle = _rle(mask_data[fidx][oid])
            # Absent objects are all-zero RLEs; skip them by area (see data_qwen.py).
            if rle is None or maskUtils.area(rle) == 0:
                continue
            m = maskUtils.decode(rle)
            if m.ndim == 3:
                m = m[:, :, 0]
            m = torch.from_numpy(m.astype(np.uint8))[None, None].float()
            out[o, n] = (F.interpolate(m, size=(h_rz, w_rz), mode="nearest")[0, 0] > 0.5).to(torch.uint8)
    return out


# --------------------------------------------------------------------------- targets
def relationships_in_seconds(rels, fps: float, clip_start_sec: int, sec_lo: int, sec_hi: int,
                             id_map: Dict[int, int]) -> List[list]:
    """Clip relationships to whole seconds ``[sec_lo, sec_hi)`` of the clip and renumber them.

    A relation is active in second ``s`` if any of its native frames falls inside it.
    Returns ``[[s, pred, o, [[t0, t1], ...]], ...]`` with inclusive integer endpoints.
    """
    out = []
    for rel in rels:
        s, pred, o, spans = rel[0], rel[1], rel[2], rel[3]
        if s not in id_map or o not in id_map:
            continue
        secs = set()
        for f0, f1 in spans:
            a = int(np.floor(f0 / fps)) - clip_start_sec
            b = int(np.floor(f1 / fps)) - clip_start_sec
            secs.update(range(max(a, sec_lo), min(b, sec_hi - 1) + 1))
        if not secs:
            continue
        secs = sorted(secs)
        merged, start, prev = [], secs[0], secs[0]
        for t in secs[1:]:
            if t != prev + 1:
                merged.append([start, prev])
                start = t
            prev = t
        merged.append([start, prev])
        out.append([id_map[s], pred, id_map[o], merged])
    out.sort(key=lambda r: (r[3][0][0], r[0], r[2], r[1]))
    return out


@dataclass
class ClipSpec:
    item: int
    t_sec: int       # anchor: last observed second boundary (clip time H)
    history: int     # H, seconds
    horizon: int     # F, seconds


def clip_frames(item, spec: ClipSpec, sample_fps: int, window_sec: int):
    """Native frame index of every sampled frame of the clip, and the frames of every window."""
    fps = float(item["fps"])
    start = spec.t_sec - spec.history
    n = (spec.history + spec.horizon) * sample_fps
    times = start + (np.arange(n) + 0.5) / sample_fps
    frame_idx = np.clip(np.round(times * fps).astype(int), 0, int(item["num_frames"]) - 1).tolist()
    per_win = sample_fps * window_sec
    return frame_idx, [frame_idx[k:k + per_win] for k in range(0, n, per_win)]


def clip_targets(item, spec: ClipSpec, mask_data, args) -> Dict:
    """Objects of the clip, their per-window boxes and the future relationships.

    Objects: those visible in the observed seconds (the future cannot introduce new ones),
    optionally only those taking part in some relationship of the video. Shared by the
    dataset, the baselines and the evaluation so that all of them score the same clips.
    """
    H, Fh = spec.history, spec.horizon
    n_hist = H // int(args.window_sec)
    frame_idx, windows = clip_frames(item, spec, int(args.sample_fps), int(args.window_sec))
    rels = item.get("relationships", [])
    obj_ids = [o["id"] for o in item["objects"]]
    names = {o["id"]: o["name"] for o in item["objects"]}
    if args.object_filter == "related":
        related = {r[0] for r in rels} | {r[2] for r in rels}
        obj_ids = [o for o in obj_ids if o in related]
    boxes, valid = window_boxes(mask_data, obj_ids, windows)
    keep = [k for k in range(len(obj_ids)) if bool(valid[k, :n_hist].any())][: int(args.max_objects)]
    obj_ids = [obj_ids[k] for k in keep]
    boxes, valid = boxes[keep], valid[keep]
    id_map = {oid: k + 1 for k, oid in enumerate(obj_ids)}  # "Object k" numbering
    start = spec.t_sec - H
    return dict(
        frame_idx=frame_idx, n_hist_windows=n_hist, n_fut_windows=Fh // int(args.window_sec),
        obj_ids=obj_ids, obj_names=[names[o] for o in obj_ids], id_map=id_map,
        boxes=boxes, box_valid=valid,
        target=relationships_in_seconds(rels, float(item["fps"]), start, H, H + Fh, id_map),
        observed=relationships_in_seconds(rels, float(item["fps"]), start, 0, H, id_map),
    )


# --------------------------------------------------------------------------- dataset
class ForecastDataset(Dataset):
    """Clips for scene graph forecasting.

    In training mode every ``__getitem__`` draws a random anchor, history and horizon for
    video ``i`` (biased towards anchors with relations in the future). In evaluation mode
    the clips are enumerated once: every ``eval_stride`` seconds, fixed ``history`` /
    ``horizon``.
    """

    def __init__(self, annotation_path: str, processor, tokenizer, args, train: bool = True):
        self.items = load_annotations(annotation_path)
        self.processor = processor
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.chat_template = CHAT_TEMPLATE
        self.args = args
        self.train = train
        self.sample_fps = int(args.sample_fps)
        self.window_sec = int(args.window_sec)
        assert (self.sample_fps * self.window_sec) % 2 == 0, \
            "sample_fps * window_sec must be even: a window is made of whole 2-frame temporal grids"
        self.clips: List[ClipSpec] = [] if train else self._enumerate_clips()

    # ---- clip selection ----
    def _duration(self, item) -> int:
        return int(item["num_frames"] / item["fps"])

    def _event_anchors(self, item, horizon: int) -> List[int]:
        """Anchors whose future window contains the start of a relationship."""
        starts = {int(span[0] / item["fps"]) for rel in item.get("relationships", []) for span in rel[3]}
        return sorted({s - d for s in starts for d in range(horizon)})

    def _enumerate_clips(self) -> List[ClipSpec]:
        H, Fh = int(self.args.history_sec), int(self.args.horizon_sec)
        clips = []
        for i, item in enumerate(self.items):
            dur = self._duration(item)
            for t in range(H, dur - Fh + 1, int(self.args.eval_stride)):
                clips.append(ClipSpec(i, t, H, Fh))
        return clips

    def _random_clip(self, i: int) -> ClipSpec:
        item = self.items[i]
        a = self.args
        H = random.randint(int(a.history_min_sec), int(a.history_max_sec))
        Fh = random.randint(int(a.horizon_min_sec), int(a.horizon_max_sec))
        dur = self._duration(item)
        lo, hi = H, dur - Fh
        if hi < lo:
            raise ValueError(f"video {item['video']} ({dur}s) is shorter than H+F={H + Fh}s")
        events = [t for t in self._event_anchors(item, Fh) if lo <= t <= hi]
        if events and random.random() < float(a.event_anchor_prob):
            t = random.choice(events)
        else:
            t = random.randint(lo, hi)
        return ClipSpec(i, t, H, Fh)

    def __len__(self):
        return len(self.items) if self.train else len(self.clips)

    def __getitem__(self, i):
        for attempt in range(10):
            try:
                spec = self._random_clip(i) if self.train else self.clips[i]
                return self.build(spec)
            except SkipClip:
                if not self.train:
                    raise
                i = random.randrange(len(self.items))
        raise RuntimeError("could not build a training clip after 10 attempts")

    # ---- sample construction ----
    def build(self, spec: ClipSpec) -> Dict:
        item = self.items[spec.item]
        a = self.args
        H, Fh = spec.history, spec.horizon
        with open(item["mask_json"]) as f:
            mask_data = json.load(f)
        ct = clip_targets(item, spec, mask_data, a)
        if not ct["obj_ids"]:
            raise SkipClip("no object visible in the observed seconds")
        obj_ids, frame_idx = ct["obj_ids"], ct["frame_idx"]
        n_hist_win, n_fut_win = ct["n_hist_windows"], ct["n_fut_windows"]
        all_boxes, all_valid = ct["boxes"], ct["box_valid"]

        # Video.
        frames = read_frames(item["video"], frame_idx)
        vp = copy.deepcopy(self.processor.video_processor)
        vp.max_pixels = a.video_max_frame_pixels
        vp.min_pixels = a.video_min_frame_pixels
        vp.size["longest_edge"] = vp.max_pixels
        vp.size["shortest_edge"] = vp.min_pixels
        out = vp.preprocess(videos=frames, return_tensors="pt")
        grid = out["video_grid_thw"][0]
        patch = vp.patch_size
        obj_masks = decode_masks(mask_data, obj_ids, frame_idx, int(grid[1]) * patch, int(grid[2]) * patch)

        # Text.
        target_rels = ct["target"]
        if self.train and not target_rels and random.random() > float(a.keep_empty_prob):
            raise SkipClip("no future relationships")
        prompt = FORECAST_PROMPT.format(F=Fh, H=H, E=H + Fh).replace("<video>", VIDEO_PLACEHOLDER)
        answer = json.dumps({"relationships": target_rels}, ensure_ascii=False)
        input_ids, labels = self._tokenize(prompt, answer)

        labels_txt = [
            f"Object {k + 1} ({name}): " if a.object_names_in_prompt else f"Object {k + 1}: "
            for k, name in enumerate(ct["obj_names"])
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values_videos=out["pixel_values_videos"],
            video_grid_thw=grid[None],
            obj_masks=obj_masks,
            obj_labels=labels_txt,
            n_hist_windows=n_hist_win,
            n_fut_windows=n_fut_win,
            hist_boxes=all_boxes[:, :n_hist_win],
            hist_box_valid=all_valid[:, :n_hist_win],
            fut_boxes=all_boxes[:, n_hist_win:],
            fut_box_valid=all_valid[:, n_hist_win:],
            meta=dict(video=item["video"], item=spec.item, t_sec=spec.t_sec, history=H, horizon=Fh,
                      obj_ids=obj_ids, obj_names=ct["obj_names"], target=target_rels,
                      observed=ct["observed"]),
        )

    def _tokenize(self, prompt: str, answer: Optional[str]):
        tok = self.tokenizer
        ids = list(tok.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}], return_dict=False))
        if answer is None:
            ids += list(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, return_dict=False))
            return torch.tensor(ids), None
        labels = [IGNORE_INDEX] * len(ids)
        user = list(tok.apply_chat_template([{"role": "user", "content": prompt}], return_dict=False))
        ids += user
        labels += [IGNORE_INDEX] * len(user)
        asst = list(tok.apply_chat_template([{"role": "assistant", "content": answer}], return_dict=False))
        ids += asst
        labels += [IGNORE_INDEX] * 3 + asst[3:]  # mask "<|im_start|>assistant\n", as in data_qwen.py
        return torch.tensor(ids), torch.tensor(labels)

    def generation_prompt_ids(self, spec_or_sample) -> torch.Tensor:
        H, Fh = spec_or_sample["meta"]["history"], spec_or_sample["meta"]["horizon"]
        prompt = FORECAST_PROMPT.format(F=Fh, H=H, E=H + Fh).replace("<video>", VIDEO_PLACEHOLDER)
        ids, _ = self._tokenize(prompt, None)
        return ids


class SkipClip(Exception):
    pass


@dataclass
class ForecastCollator:
    pad_token_id: int

    _list_keys = ("obj_masks", "obj_labels", "n_hist_windows", "n_fut_windows", "hist_boxes",
                  "hist_box_valid", "fut_boxes", "fut_box_valid", "meta")

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        ids = [x["input_ids"] for x in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b, x in enumerate(ids):
            attention_mask[b, : len(x)] = True
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=torch.cat([x["pixel_values_videos"] for x in instances], dim=0),
            video_grid_thw=torch.cat([x["video_grid_thw"] for x in instances], dim=0),
        )
        if instances[0]["labels"] is not None:
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                [x["labels"] for x in instances], batch_first=True, padding_value=IGNORE_INDEX)
        for k in self._list_keys:
            batch[k] = [x[k] for x in instances]
        return batch
