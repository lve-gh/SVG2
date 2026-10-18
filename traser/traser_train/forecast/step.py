"""Batch -> arranged model inputs. Shared by training (trainer_forecast.py) and evaluation."""

from typing import Dict

import torch

from ..train.token_selection import select_tokens
from .arrangement import arrange_forecast

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


def _select(masks: torch.Tensor, grid, coverage_thresh, time_reduce, ensure_at_least_one):
    _, idx, _ = select_tokens(
        obj_masks=masks.float(), grid_thw=grid, patch_size=PATCH_SIZE, spatial_merge_size=SPATIAL_MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE, coverage_thresh=coverage_thresh, time_reduce=time_reduce,
        device="cpu", ensure_at_least_one=ensure_at_least_one)
    return idx


def _tok(tokenizer, texts):
    if not texts:
        return []
    enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    return [torch.tensor(x, dtype=torch.long) for x in enc]


def build_forecast_inputs(base_model, tokenizer, batch: Dict, *, sample_fps: int, window_sec: int,
                          coverage_thresh: float = 0.5, time_reduce: str = "max", mode: str = "latent",
                          with_teacher: bool = True):
    """Token selection (history / future grids separately) + arrangement with forecast slots."""
    frames_per_window = sample_fps * window_sec
    grids_per_window = frames_per_window // TEMPORAL_PATCH_SIZE
    B = batch["input_ids"].shape[0]
    hist_idx, fut_idx, label_ids, hist_ts, fut_ts = [], [], [], [], []
    for b in range(B):
        masks = batch["obj_masks"][b]                                   # (O, N, H, W) uint8
        T, Hp, Wp = (int(x) for x in batch["video_grid_thw"][b])
        Wh, Wf = int(batch["n_hist_windows"][b]), int(batch["n_fut_windows"][b])
        Nh = Wh * frames_per_window
        Th = Nh // TEMPORAL_PATCH_SIZE
        # History indices are numbered from the clip start, so they index the full-clip features.
        hist_idx.append(_select(masks[:, :Nh], (Th, Hp, Wp), coverage_thresh, time_reduce, True))
        if with_teacher and masks.shape[1] > Nh:
            per_grid = (Hp // SPATIAL_MERGE_SIZE) * (Wp // SPATIAL_MERGE_SIZE)
            f_idx = _select(masks[:, Nh:], (T - Th, Hp, Wp), coverage_thresh, time_reduce, False)
            fut_idx.append([i + Th * per_grid for i in f_idx])
        label_ids.append(_tok(tokenizer, batch["obj_labels"][b]))
        hist_ts.append(_tok(tokenizer, [f"<{w * window_sec} - {(w + 1) * window_sec} sec>" for w in range(Wh)]))
        fut_ts.append(_tok(tokenizer, [f"<forecast {(Wh + f) * window_sec} - {(Wh + f + 1) * window_sec} sec>"
                                       for f in range(Wf)]))

    dev = batch["input_ids"].device
    return arrange_forecast(
        model=base_model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values_videos=batch["pixel_values_videos"],
        video_grid_thw=batch["video_grid_thw"],
        hist_obj_idx=hist_idx,
        obj_label_ids=label_ids,
        n_hist_windows=[int(x) for x in batch["n_hist_windows"]],
        n_fut_windows=[int(x) for x in batch["n_fut_windows"]],
        grids_per_window=grids_per_window,
        hist_ts_ids=hist_ts,
        fut_ts_ids=fut_ts,
        hist_boxes=[x.to(dev) for x in batch["hist_boxes"]],
        hist_box_valid=[x.to(dev) for x in batch["hist_box_valid"]],
        fut_obj_idx=fut_idx if (with_teacher and fut_idx) else None,
        labels=batch.get("labels"),
        mode=mode,
        obj_traj_start_id=getattr(base_model.config, "obj_traj_start_id", None),
        obj_traj_end_id=getattr(base_model.config, "obj_traj_end_id", None),
    )
