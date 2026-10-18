"""Trajectory-aligned token arrangement with forecast slots.

Extends TRASER's object blocks (``train/token_arrangement.py``) with ``F`` future windows
per object whose latents come from the forecaster instead of from video tokens:

    <obj_traj_start> Object k (name): <|vision_start|>
        [OTR latents]                                    history tokens only
        <0 - 1 sec> [TWR latents] ... <H-1 - H sec> [TWR latents]
        <forecast H - H+1 sec> [future latents] ... <forecast H+F-1 - H+F sec> [future latents]
    <|vision_end|> <obj_traj_end>

The vision encoder runs on the whole clip (history + future). Qwen2.5-VL attends only within
one temporal grid, so the history features are identical to an encoding of the history
alone: no future information leaks into the LLM input. Future-window features are used only
for the latent-distillation targets (no gradient).

``mode`` selects what fills the future slots:
    latent   forecaster predictions (TRASER-F)
    copy     the object's last observed TWR latents (persistence ablation)
    oracle   TWR latents of the real future frames (upper bound; needs future frames)
    prompt   no future slots: the LLM forecasts from the history blocks alone
"""

from typing import Dict, List, Optional

import torch

PAD_ID = 151643  # Qwen2.5-VL <|endoftext|>
MODES = ("latent", "copy", "oracle", "prompt")


def _video_features(model, pixel_values_videos, video_grid_thw) -> List[torch.Tensor]:
    feats = model.model.get_video_features(pixel_values_videos.type(model.model.visual.dtype), video_grid_thw)
    if hasattr(feats, "pooler_output"):  # transformers >= 5
        feats = feats.pooler_output
    if isinstance(feats, (list, tuple)):
        return list(feats)
    counts = [int(t * h * w) // 4 for t, h, w in video_grid_thw.tolist()]
    return list(torch.split(feats, counts, dim=0))


def _resample(resampler, seqs: List[torch.Tensor]) -> torch.Tensor:
    x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    lens = torch.tensor([s.shape[0] for s in seqs], device=x.device)
    mask = torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]
    return resampler(x, attention_mask=mask)


def arrange_forecast(
    model,
    input_ids: torch.LongTensor,                 # [B, L], one <|video_pad|> between vision start/end
    attention_mask: torch.Tensor,                # [B, L]
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.LongTensor,            # [B, 3]
    hist_obj_idx: List[List[torch.Tensor]],      # per sample, per object: token indices in history grids
    obj_label_ids: List[List[torch.Tensor]],     # "Object k (name): " ids
    n_hist_windows: List[int],
    n_fut_windows: List[int],
    grids_per_window: int,
    hist_ts_ids: List[List[torch.Tensor]],       # per sample, per history window
    fut_ts_ids: List[List[torch.Tensor]],        # per sample, per future window
    hist_boxes: Optional[List[torch.Tensor]] = None,
    hist_box_valid: Optional[List[torch.Tensor]] = None,
    fut_obj_idx: Optional[List[List[torch.Tensor]]] = None,   # future-grid token indices (teacher)
    labels: Optional[torch.LongTensor] = None,
    mode: str = "latent",
    obj_traj_start_id: Optional[int] = None,
    obj_traj_end_id: Optional[int] = None,
    ignore_id: int = -100,
):
    assert mode in MODES, mode
    dev = input_ids.device
    B = input_ids.shape[0]
    cfg = model.config
    vt_id, vs_id, ve_id = int(cfg.video_token_id), int(cfg.vision_start_token_id), int(cfg.vision_end_token_id)
    twr, otr = model.perceiver_resampler, getattr(model, "second_perceiver_resampler", None)
    R = int(twr.n_latents)
    R2 = int(otr.n_latents) if otr is not None else 0
    use_future_slots = mode != "prompt"

    feats = _video_features(model, pixel_values_videos, video_grid_thw)

    # ---- (1) plan every sequence: token ids, labels, and where each latent group goes ----
    seq_ids, seq_labels = [], []
    otr_jobs, twr_jobs, fut_slots = [], [], []   # (b, obj, [w,] start position)
    for b in range(B):
        ids = input_ids[b][attention_mask[b].bool()].tolist()
        lab = labels[b][attention_mask[b].bool()].tolist() if labels is not None else [ignore_id] * len(ids)
        vs = ids.index(vs_id)
        ve = vs + ids[vs:].index(ve_id)
        assert all(t == vt_id for t in ids[vs + 1:ve]), "expected only <|video_pad|> inside the vision span"
        out_ids, out_lab = ids[:vs], lab[:vs]

        def put(tokens):
            out_ids.extend(tokens)
            out_lab.extend([ignore_id] * len(tokens))

        grid_tokens = int(video_grid_thw[b][1] * video_grid_thw[b][2]) // 4
        win_tokens = grid_tokens * grids_per_window
        for o, idx in enumerate(hist_obj_idx[b]):
            if obj_traj_start_id is not None:
                put([int(obj_traj_start_id)])
            put(obj_label_ids[b][o].tolist())
            put([vs_id])
            if otr is not None:
                otr_jobs.append((b, o, len(out_ids)))
                put([vt_id] * R2)
            win = idx // win_tokens
            for w in range(n_hist_windows[b]):
                if not bool((win == w).any()):
                    continue
                put(hist_ts_ids[b][w].tolist())
                twr_jobs.append((b, o, w, len(out_ids)))
                put([vt_id] * R)
            if use_future_slots:
                for f in range(n_fut_windows[b]):
                    put(fut_ts_ids[b][f].tolist())
                    fut_slots.append((b, o, f, len(out_ids)))
                    put([vt_id] * R)
            put([ve_id])
            if obj_traj_end_id is not None:
                put([int(obj_traj_end_id)])
        out_ids += ids[ve + 1:]
        out_lab += lab[ve + 1:]
        seq_ids.append(out_ids)
        seq_labels.append(out_lab)

    Lmax = max(len(s) for s in seq_ids)
    new_ids = torch.full((B, Lmax), PAD_ID, dtype=torch.long)
    new_lab = torch.full((B, Lmax), ignore_id, dtype=torch.long)
    new_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for b in range(B):
        n = len(seq_ids[b])
        new_ids[b, :n] = torch.tensor(seq_ids[b])
        new_lab[b, :n] = torch.tensor(seq_labels[b])
        new_mask[b, :n] = True
    new_ids, new_lab, new_mask = new_ids.to(dev), new_lab.to(dev), new_mask.to(dev)
    embeds = model.get_input_embeddings()(new_ids).clone()

    # ---- (2) observed latents: OTR over all history tokens, TWR per (object, window) ----
    def rows(b, idx):
        return feats[b].index_select(0, idx.to(feats[b].device))

    otr_out = {}
    if otr_jobs:
        y = _resample(otr, [rows(b, hist_obj_idx[b][o]) for b, o, _ in otr_jobs])
        for (b, o, p), lat in zip(otr_jobs, y):
            otr_out[(b, o)] = lat
            embeds[b, p:p + R2] = lat.to(embeds.dtype)

    twr_out = {}
    if twr_jobs:
        seqs = []
        for b, o, w, _ in twr_jobs:
            grid_tokens = int(video_grid_thw[b][1] * video_grid_thw[b][2]) // 4
            idx = hist_obj_idx[b][o]
            seqs.append(rows(b, idx[idx // (grid_tokens * grids_per_window) == w]))
        y = _resample(twr, seqs)
        for (b, o, w, p), lat in zip(twr_jobs, y):
            twr_out[(b, o, w)] = lat
            embeds[b, p:p + R] = lat.to(embeds.dtype)

    # ---- (3) teacher latents of the real future windows (no gradient) ----
    teacher = {}
    if fut_obj_idx is not None:
        jobs, seqs = [], []
        for b in range(B):
            grid_tokens = int(video_grid_thw[b][1] * video_grid_thw[b][2]) // 4
            win_tokens = grid_tokens * grids_per_window
            for o, idx in enumerate(fut_obj_idx[b]):
                win = idx // win_tokens - n_hist_windows[b]
                for f in range(n_fut_windows[b]):
                    sel = idx[win == f]
                    if sel.numel():
                        jobs.append((b, o, f))
                        seqs.append(rows(b, sel))
        if jobs:
            with torch.no_grad():
                y = _resample(twr, seqs)
            teacher = {k: lat for k, lat in zip(jobs, y)}

    # ---- (4) forecast, per sample (objects of one clip interact; clips do not) ----
    aux: List[Dict[str, torch.Tensor]] = []
    D = embeds.shape[-1]
    for b in range(B):
        O, Wh, Fw = len(hist_obj_idx[b]), n_hist_windows[b], n_fut_windows[b]
        hist = embeds.new_zeros(O, Wh, R, D, dtype=feats[b].dtype)
        valid = torch.zeros(O, Wh, dtype=torch.bool, device=dev)
        for o in range(O):
            for w in range(Wh):
                if (b, o, w) in twr_out:
                    hist[o, w] = twr_out[(b, o, w)]
                    valid[o, w] = True
        otr_b = torch.stack([otr_out[(b, o)] for o in range(O)]) if otr is not None else None
        out = model.forecaster(hist, valid, Fw, otr_latents=otr_b,
                               hist_boxes=hist_boxes[b].to(dev) if hist_boxes is not None else None,
                               hist_box_valid=hist_box_valid[b].to(dev) if hist_box_valid is not None else None)
        t_lat = hist.new_zeros(O, Fw, R, D)
        t_valid = torch.zeros(O, Fw, dtype=torch.bool, device=dev)
        for (bb, o, f), lat in teacher.items():
            if bb == b:
                t_lat[o, f] = lat
                t_valid[o, f] = True
        if mode == "copy":
            fill = out["copy_last_latents"][:, None].expand(O, Fw, R, D)
        elif mode == "oracle":
            assert fut_obj_idx is not None, "oracle mode needs the future frames"
            fill = torch.where(t_valid[..., None, None], t_lat, out["copy_last_latents"][:, None].expand(O, Fw, R, D))
        else:
            fill = out["future_latents"]
        out.update(teacher_latents=t_lat, teacher_valid=t_valid)
        aux.append(out)
        for bb, o, f, p in fut_slots:
            if bb == b:
                embeds[b, p:p + R] = fill[o, f].to(embeds.dtype)

    # ---- (5) positions: 1D-linear over the rearranged sequence, as in TRASER ----
    pos = torch.arange(Lmax, device=dev, dtype=torch.int32)[None].expand(B, -1)
    position_ids = pos[None].expand(3, B, Lmax).contiguous()
    lens = new_mask.sum(dim=1, keepdim=True)
    rope_deltas = (lens - Lmax).to(torch.long)  # == maxpos + 1 - Lmax for right padding
    cache_position = torch.arange(Lmax, device=dev, dtype=torch.int32)
    return embeds, position_ids, new_mask, rope_deltas, cache_position, (new_lab if labels is not None else None), aux
