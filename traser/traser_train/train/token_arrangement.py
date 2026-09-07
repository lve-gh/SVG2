from typing import List, Optional

import torch


def rearrange_token(
    model,
    input_ids: torch.LongTensor,           # [B, L]
    attention_mask: torch.LongTensor,      # [B, L]
    pixel_values: Optional[torch.FloatTensor],         # unused, kept for API compatibility
    image_grid_thw: Optional[torch.LongTensor],        # unused, kept for API compatibility
    pixel_values_videos: Optional[torch.FloatTensor],
    video_grid_thw: Optional[torch.LongTensor],
    second_per_grid_ts: Optional[torch.Tensor],

    # Per-sample list of objects; each object is a 1D LongTensor of indices into that
    # sample's video-token stream (relative indices, not absolute sequence positions).
    obj_token_indices_per_sample: List[List[torch.Tensor]],

    obj_traj_start_id: Optional[int] = None,
    obj_traj_end_id: Optional[int] = None,

    # "Object k: " label token ids: List[sample][object] -> 1D LongTensor
    text_token_ids_per_sample: Optional[List[List[torch.Tensor]]] = None,

    # Timestamp token ids per temporal window: List[sample] -> List[window] -> 1D LongTensor
    timestamp_token_ids_per_batch=None,
    # Number of grids per temporal window: List[sample] -> int
    grids_per_temporal_window_per_batch=None,

    labels: Optional[torch.LongTensor] = None,
    IGNORE_ID: int = -100,

    use_resampler: bool = True,
    use_second_resampler: bool = True,
    add_timestamp_token: bool = True,
):
    """Rearrange a Qwen2.5-VL input sequence into trajectory-aligned object blocks.

    The original contiguous video-token span is replaced by one block per object:

        <obj_traj_start> Object k: <|vision_start|>
            [OTR latents]                      (global object summary)
            <t0 - t1 sec> [TWR latents]        (one group per non-empty temporal window)
            <t1 - t2 sec> [TWR latents]
            ...
        <|vision_end|> <obj_traj_end>

    Returns:
        new_inputs_embeds:  [B, Lmax, D]   (grad flows to the token embedding, the visual
                                            encoder, and the resamplers)
        new_position_ids:   [3, B, Lmax]   (int32, 1D-linear positions replicated over the 3 RoPE dims)
        new_attention_mask: [B, Lmax]
        rope_deltas:        [B, 1] (long)
        cache_position:     [Lmax] (int32)
        new_input_ids:      [B, Lmax] (long)
        new_labels:         [B, Lmax] or None (long)
    """
    dev = input_ids.device
    B, L = input_ids.shape
    cpu = torch.device("cpu")

    assert text_token_ids_per_sample is not None and len(text_token_ids_per_sample) == B, \
        "rearrange_token requires text_token_ids_per_sample with length B."
    assert grids_per_temporal_window_per_batch is not None and len(grids_per_temporal_window_per_batch) == B, \
        "grids_per_temporal_window_per_batch is required."
    if add_timestamp_token:
        assert timestamp_token_ids_per_batch is not None and len(timestamp_token_ids_per_batch) == B, \
            "add_timestamp_token=True requires timestamp_token_ids_per_batch with length B."

    tok_embed = model.get_input_embeddings()
    vt_id = int(model.config.video_token_id)
    vs_id = getattr(model.config, "vision_start_token_id", None)
    ve_id = getattr(model.config, "vision_end_token_id", None)
    pad_id = 151643  # Qwen2.5-VL <|endoftext|>

    # ---- (0) Temporal window metadata ----
    assert video_grid_thw is not None, "video_grid_thw is required for temporal windowing"
    assert video_grid_thw.shape[0] == B and video_grid_thw.shape[1] == 3, \
        f"video_grid_thw should be ({B},3), got {video_grid_thw.shape}"

    grid_area_batch: List[int] = []  # spatial tokens per temporal grid, after 2x2 spatial merging
    temporal_window_size_batch = grids_per_temporal_window_per_batch

    # ---- (1) Compute visual features (with grad) ----
    video_embeds = None
    if pixel_values_videos is not None:
        _vid = model.model.get_video_features(
            pixel_values_videos.type(model.model.visual.dtype), video_grid_thw
        )
        if hasattr(_vid, "pooler_output"):  # transformers >= 5 wraps the features in a ModelOutput
            _vid = _vid.pooler_output
        video_embeds = torch.cat(_vid, dim=0) if isinstance(_vid, (list, tuple)) else _vid  # [N_vid, D]
        del pixel_values_videos, _vid

    # ---- (2) Resampler handles ----
    resampler = None
    resampler_num_latents = None
    second_resampler = None
    second_resampler_num_latents = None
    if use_resampler:
        if not hasattr(model, "perceiver_resampler"):
            raise RuntimeError("use_resampler=True, but model.perceiver_resampler not found.")
        resampler = model.perceiver_resampler
        resampler_num_latents = int(resampler.n_latents)
        if use_second_resampler:
            if not hasattr(model, "second_perceiver_resampler"):
                raise RuntimeError("use_second_resampler=True, but model.second_perceiver_resampler not found.")
            second_resampler = model.second_perceiver_resampler
            second_resampler_num_latents = int(second_resampler.n_latents)

    # ---- (3) Move to CPU for sequence planning (no autograd graph) ----
    attn_cpu = attention_mask.to(cpu, dtype=torch.bool)
    ids_cpu = input_ids.to(cpu)
    lbls_cpu = labels.to(cpu) if labels is not None else None

    # Effective lengths and video-token positions in the original sequence.
    eff_lens: List[int] = []
    vid_idx_list: List[torch.Tensor] = []
    for b in range(B):
        video_grid_thw_b = video_grid_thw[b]
        grid_area = (int(video_grid_thw_b[1].item()) * int(video_grid_thw_b[2].item())) // 4
        grid_area_batch.append(int(grid_area))

        nz = torch.nonzero(attn_cpu[b], as_tuple=False).flatten()
        L_eff = int(nz[-1].item()) + 1 if nz.numel() > 0 else 0
        eff_lens.append(L_eff)

        if L_eff > 0:
            ids_b_eff = ids_cpu[b, :L_eff]
            vid_idx = torch.nonzero(ids_b_eff == vt_id, as_tuple=False).flatten()
            vid_idx_list.append(vid_idx)
        else:
            vid_idx_list.append(torch.empty(0, dtype=torch.long))

    # Row offsets of each sample's video tokens inside the concatenated video_embeds.
    vid_counts = [int(v.numel()) for v in vid_idx_list]
    vid_offsets: List[int] = [0] * B
    running = 0
    for b in range(B):
        vid_offsets[b] = running
        running += vid_counts[b]

    # ---- (4) Length planning ----
    def _object_block_len(b: int, obj_i: int, sel_latent_len: int, rel_temporal_window_idx: torch.Tensor) -> int:
        """Total sequence length of one object block."""
        add = 0

        if obj_traj_start_id is not None:
            add += 1

        add += int(text_token_ids_per_sample[b][obj_i].numel())

        if vs_id is not None:
            add += 1

        if add_timestamp_token and timestamp_token_ids_per_batch is not None:
            locs = rel_temporal_window_idx.unique()
            for loc in locs:
                loc_i = int(loc.item())
                if loc_i < len(timestamp_token_ids_per_batch[b]):
                    add += int(timestamp_token_ids_per_batch[b][loc_i].numel())
                else:
                    add += int(timestamp_token_ids_per_batch[b][-1].numel())

        add += int(sel_latent_len)

        if ve_id is not None:
            add += 1

        if obj_traj_end_id is not None:
            add += 1

        return add

    L_new_each: List[int] = []

    for b in range(B):
        L_eff = eff_lens[b]
        ids_b = ids_cpu[b, :L_eff]
        vid_idx = vid_idx_list[b]

        if L_eff == 0:
            L_new_each.append(0)
            continue
        if vid_idx.numel() == 0:
            L_new_each.append(L_eff)
            continue

        # Span [v_s, v_e] of the original video segment, including <|vision_start|>/<|vision_end|>.
        v_s = int(vid_idx[0].item())
        v_e = int(vid_idx[-1].item())
        has_vs = (vs_id is not None and v_s - 1 >= 0 and ids_b[v_s - 1].item() == vs_id)
        has_ve = (ve_id is not None and v_e + 1 < L_eff and ids_b[v_e + 1].item() == ve_id)
        if has_vs:
            v_s -= 1
        if has_ve:
            v_e += 1

        prefix_len = v_s
        suffix_len = L_eff - (v_e + 1)

        sel_lists = obj_token_indices_per_sample[b]

        cur_total = 0
        for i, rel in enumerate(sel_lists):
            rel = rel.to(cpu, dtype=torch.long)

            tokens_per_window = int(grid_area_batch[b] * int(temporal_window_size_batch[b]))
            rel_temporal_window_idx = rel // tokens_per_window if (tokens_per_window > 0) else torch.zeros_like(rel)
            nonempty_windows = int(rel_temporal_window_idx.unique().numel())

            # Each object contributes a fixed number of latents:
            # OTR (global summary) + TWR (one group per non-empty temporal window).
            if use_second_resampler and second_resampler_num_latents is not None:
                sel_len = int(second_resampler_num_latents) + int(resampler_num_latents) * nonempty_windows
            else:
                sel_len = int(resampler_num_latents) * nonempty_windows

            cur_total += _object_block_len(b, i, sel_len, rel_temporal_window_idx)

        L_new_each.append(prefix_len + cur_total + suffix_len)

    Lmax = max(L_new_each) if len(L_new_each) > 0 else 0

    # ---- (5) Allocate new sequence tensors on CPU and fill per sample ----
    new_input_ids_cpu = torch.full((B, Lmax), pad_id, dtype=torch.long, device=cpu)
    new_attention_mask_cpu = torch.zeros((B, Lmax), dtype=attn_cpu.dtype, device=cpu)
    new_position_ids_cpu = torch.zeros((3, B, Lmax), dtype=torch.int32, device=cpu)
    new_labels_cpu = None
    if labels is not None:
        new_labels_cpu = torch.full((B, Lmax), IGNORE_ID, dtype=torch.long, device=cpu)

    # Per-object plan for the batched resampler forwards.
    batched_obj_rows: List[torch.Tensor] = []   # rows into video_embeds for one (object, window)
    batched_obj_pos: List[torch.Tensor] = []    # destination positions in the new sequence
    batched_obj_bids: List[int] = []            # sample index
    batched_obj_lens: List[int] = []            # number of input rows (before resampling)

    batched_second_rows: List[torch.Tensor] = []
    batched_second_pos: List[torch.Tensor] = []
    batched_second_bids: List[int] = []

    def _text_pos_block(start_scalar: int, length: int, dtype=torch.int32) -> torch.Tensor:
        """1D-linear positions replicated across the 3 RoPE dims: [[i..], [i..], [i..]]."""
        if length <= 0:
            return torch.empty(3, 0, dtype=dtype, device=cpu)
        ar = torch.arange(start_scalar, start_scalar + length, device=cpu, dtype=dtype)
        return torch.stack([ar, ar, ar], dim=0)

    for b in range(B):
        L_eff = eff_lens[b]
        if L_eff == 0:
            continue

        ids_b = ids_cpu[b, :L_eff]
        msk_b = attn_cpu[b, :L_eff]
        labs_b = lbls_cpu[b, :L_eff] if lbls_cpu is not None else None
        vid_idx = vid_idx_list[b]

        dst = 0  # write cursor into the new_* tensors

        if vid_idx.numel() == 0:
            # No video segment: copy the sample as-is.
            new_input_ids_cpu[b, :L_eff] = ids_b
            new_attention_mask_cpu[b, :L_eff] = msk_b
            if new_labels_cpu is not None and labs_b is not None:
                new_labels_cpu[b, :L_eff] = labs_b
            new_position_ids_cpu[:, b, :L_eff] = _text_pos_block(0, L_eff, dtype=torch.int32)
            continue

        v_s = int(vid_idx[0].item())
        v_e = int(vid_idx[-1].item())
        has_vs = (vs_id is not None and v_s - 1 >= 0 and ids_b[v_s - 1].item() == vs_id)
        has_ve = (ve_id is not None and v_e + 1 < L_eff and ids_b[v_e + 1].item() == ve_id)
        if has_vs:
            v_s -= 1
        if has_ve:
            v_e += 1

        prefix_len = v_s
        suffix_len = L_eff - (v_e + 1)

        # Prefix before the video segment.
        if prefix_len > 0:
            new_input_ids_cpu[b, dst:dst + prefix_len] = ids_b[:prefix_len]
            new_attention_mask_cpu[b, dst:dst + prefix_len] = msk_b[:prefix_len]
            if new_labels_cpu is not None and labs_b is not None:
                new_labels_cpu[b, dst:dst + prefix_len] = labs_b[:prefix_len]
            new_position_ids_cpu[:, b, dst:dst + prefix_len] = _text_pos_block(dst, prefix_len, dtype=torch.int32)
            dst += prefix_len

        Nv = int(vid_idx.numel())
        vid_offset = int(vid_offsets[b])

        sel_lists = obj_token_indices_per_sample[b]
        for i, rel in enumerate(sel_lists):
            rel = rel.to(cpu, dtype=torch.long)
            if rel.numel() > 0:
                rel.clamp_(0, Nv - 1)

            # (a) <obj_traj_start>
            if obj_traj_start_id is not None:
                new_input_ids_cpu[b, dst] = int(obj_traj_start_id)
                new_position_ids_cpu[:, b, dst:dst + 1] = _text_pos_block(dst, 1, dtype=torch.int32)
                if new_labels_cpu is not None:
                    new_labels_cpu[b, dst] = IGNORE_ID
                new_attention_mask_cpu[b, dst] = True
                dst += 1

            # (b) "Object k: " label tokens
            txt_ids = text_token_ids_per_sample[b][i].to(cpu, dtype=torch.long)
            k = int(txt_ids.numel())
            if k > 0:
                new_input_ids_cpu[b, dst:dst + k] = txt_ids
                new_position_ids_cpu[:, b, dst:dst + k] = _text_pos_block(dst, k, dtype=torch.int32)
                if new_labels_cpu is not None:
                    new_labels_cpu[b, dst:dst + k] = IGNORE_ID
                new_attention_mask_cpu[b, dst:dst + k] = True
                dst += k

            # (c) <|vision_start|>
            if vs_id is not None:
                new_input_ids_cpu[b, dst] = int(vs_id)
                new_position_ids_cpu[:, b, dst:dst + 1] = _text_pos_block(dst, 1, dtype=torch.int32)
                if new_labels_cpu is not None:
                    new_labels_cpu[b, dst] = IGNORE_ID
                new_attention_mask_cpu[b, dst] = True
                dst += 1

            # (d) resampled visual tokens
            if rel.numel() > 0:
                tokens_per_window = int(grid_area_batch[b] * int(temporal_window_size_batch[b]))
                rel_temporal_window_idx = rel // tokens_per_window if (tokens_per_window > 0) else torch.zeros_like(rel)
                W_eff = int(rel_temporal_window_idx.max().item()) + 1 if rel_temporal_window_idx.numel() > 0 else 0

                # OTR: reserve slots for the global object summary.
                if use_second_resampler and second_resampler is not None:
                    rows_all = rel + vid_offset

                    if rows_all.numel() > 0:
                        R2 = int(second_resampler_num_latents)
                        new_input_ids_cpu[b, dst:dst + R2] = int(vt_id)
                        new_position_ids_cpu[:, b, dst:dst + R2] = _text_pos_block(dst, R2, dtype=torch.int32)
                        if new_labels_cpu is not None:
                            new_labels_cpu[b, dst:dst + R2] = IGNORE_ID
                        new_attention_mask_cpu[b, dst:dst + R2] = True

                        batched_second_rows.append(rows_all)
                        batched_second_pos.append(torch.arange(dst, dst + R2, dtype=torch.long, device=cpu))
                        batched_second_bids.append(b)

                        dst += R2

                # TWR: one timestamp + latent group per non-empty temporal window.
                R = int(resampler_num_latents)
                for w in range(W_eff):
                    m_w = (rel_temporal_window_idx == w)
                    if not torch.any(m_w):
                        continue

                    if add_timestamp_token and (timestamp_token_ids_per_batch is not None):
                        if w < len(timestamp_token_ids_per_batch[b]):
                            ts_ids = timestamp_token_ids_per_batch[b][w].to(cpu, dtype=torch.long)
                        else:
                            ts_ids = timestamp_token_ids_per_batch[b][-1].to(cpu, dtype=torch.long)
                        kt = int(ts_ids.numel())
                        assert kt > 0, "Timestamp token ids should not be empty."

                        new_input_ids_cpu[b, dst:dst + kt] = ts_ids
                        new_position_ids_cpu[:, b, dst:dst + kt] = _text_pos_block(dst, kt, dtype=torch.int32)
                        if new_labels_cpu is not None:
                            new_labels_cpu[b, dst:dst + kt] = IGNORE_ID
                        new_attention_mask_cpu[b, dst:dst + kt] = True
                        dst += kt

                    new_input_ids_cpu[b, dst:dst + R] = int(vt_id)
                    new_position_ids_cpu[:, b, dst:dst + R] = _text_pos_block(dst, R, dtype=torch.int32)
                    if new_labels_cpu is not None:
                        new_labels_cpu[b, dst:dst + R] = IGNORE_ID
                    new_attention_mask_cpu[b, dst:dst + R] = True

                    rows_w = rel[m_w] + vid_offset

                    batched_obj_rows.append(rows_w)
                    batched_obj_pos.append(torch.arange(dst, dst + R, dtype=torch.long, device=cpu))
                    batched_obj_bids.append(b)
                    batched_obj_lens.append(int(rows_w.numel()))

                    dst += R

            # (e) <|vision_end|>
            if ve_id is not None:
                new_input_ids_cpu[b, dst] = int(ve_id)
                new_position_ids_cpu[:, b, dst:dst + 1] = _text_pos_block(dst, 1, dtype=torch.int32)
                if new_labels_cpu is not None:
                    new_labels_cpu[b, dst] = IGNORE_ID
                new_attention_mask_cpu[b, dst] = True
                dst += 1

            # (f) <obj_traj_end>
            if obj_traj_end_id is not None:
                new_input_ids_cpu[b, dst] = int(obj_traj_end_id)
                new_position_ids_cpu[:, b, dst:dst + 1] = _text_pos_block(dst, 1, dtype=torch.int32)
                if new_labels_cpu is not None:
                    new_labels_cpu[b, dst] = IGNORE_ID
                new_attention_mask_cpu[b, dst] = True
                dst += 1

        # Suffix after the video segment.
        if suffix_len > 0:
            src_lo = v_e + 1
            src_hi = L_eff
            seg = src_hi - src_lo
            new_input_ids_cpu[b, dst:dst + seg] = ids_b[src_lo:src_hi]
            new_attention_mask_cpu[b, dst:dst + seg] = msk_b[src_lo:src_hi]
            if new_labels_cpu is not None and labs_b is not None:
                new_labels_cpu[b, dst:dst + seg] = labs_b[src_lo:src_hi]
            new_position_ids_cpu[:, b, dst:dst + seg] = _text_pos_block(dst, seg, dtype=torch.int32)
            dst += seg

        assert dst == L_new_each[b], f"sample {b}: dst={dst}, L_new={L_new_each[b]}"

    # ---- (6) Move back to device and build inputs_embeds ----
    new_input_ids = new_input_ids_cpu.to(dev, non_blocking=True)
    new_position_ids = new_position_ids_cpu.to(dev, non_blocking=True)
    new_attention_mask = new_attention_mask_cpu.to(dev, non_blocking=True)
    new_labels = None if new_labels_cpu is None else new_labels_cpu.to(dev, non_blocking=True)

    base = tok_embed(new_input_ids)  # [B, Lmax, D], with graph
    new_inputs_embeds = base.clone()

    # ---- (6.1) OTR forward: object-level global summaries ----
    if use_resampler and use_second_resampler and len(batched_second_rows) > 0:
        if video_embeds is None:
            raise RuntimeError("use_second_resampler=True but video_embeds is None.")
        dev_emb = video_embeds.device
        dtype_emb = video_embeds.dtype
        D = video_embeds.shape[-1]
        N_obj2 = len(batched_second_rows)

        seqs2 = []
        lens2 = []
        for rows_all in batched_second_rows:
            if rows_all.numel() == 0:
                seqs2.append(torch.zeros(0, D, device=dev_emb, dtype=dtype_emb))
                lens2.append(0)
            else:
                seqs2.append(video_embeds.index_select(0, rows_all.to(dev_emb)))
                lens2.append(int(rows_all.numel()))
        x2 = torch.nn.utils.rnn.pad_sequence(seqs2, batch_first=True)  # [N_obj2, L2_max, D]
        L2_max = x2.size(1)
        lens2_t = torch.tensor(lens2, device=dev_emb, dtype=torch.long)
        ar2 = torch.arange(L2_max, device=dev_emb).unsqueeze(0)
        mask2 = (ar2 < lens2_t.unsqueeze(1))  # [N_obj2, L2_max]

        y2 = second_resampler(x2, attention_mask=mask2)  # [N_obj2, R2, D]
        y2 = y2.to(new_inputs_embeds.dtype)

        for j in range(N_obj2):
            b_cur = batched_second_bids[j]
            pos2 = batched_second_pos[j].to(dev)
            new_inputs_embeds[b_cur, pos2] = y2[j]

    # ---- (6.2) TWR forward: per-window groups, all objects batched together ----
    if use_resampler and len(batched_obj_rows) > 0:
        if video_embeds is None:
            raise RuntimeError("use_resampler=True but video_embeds is None.")
        dev_emb = video_embeds.device
        dtype_emb = video_embeds.dtype
        D = video_embeds.shape[-1]

        N_obj = len(batched_obj_rows)
        lens = torch.tensor(batched_obj_lens, device=dev_emb, dtype=torch.long)
        L_max = int(lens.max().item())

        seqs = []
        for rows in batched_obj_rows:
            if rows.numel() == 0:
                seqs.append(torch.zeros(0, D, device=dev_emb, dtype=dtype_emb))
            else:
                seqs.append(video_embeds.index_select(0, rows.to(dev_emb)))
        x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # [N_obj, L_max, D]

        ar = torch.arange(L_max, device=dev_emb).unsqueeze(0)
        mask = (ar < lens.unsqueeze(1))  # [N_obj, L_max]

        y = resampler(x, attention_mask=mask)  # [N_obj, R, D]
        y = y.to(new_inputs_embeds.dtype)

        # Scatter back, merging per-sample indices to reduce kernel launches.
        per_b_indices: List[List[int]] = [[] for _ in range(B)]
        for i in range(N_obj):
            per_b_indices[batched_obj_bids[i]].append(i)

        for b in range(B):
            if not per_b_indices[b]:
                continue
            pos_list = []
            emb_list = []
            for i in per_b_indices[b]:
                pos_list.append(batched_obj_pos[i].to(dev))
                emb_list.append(y[i])
            pos_b = torch.cat(pos_list, dim=0)
            emb_b = torch.cat(emb_list, dim=0)
            new_inputs_embeds[b, pos_b] = emb_b

    # ---- (7) rope_deltas / cache_position ----
    maxpos = new_position_ids.max(dim=0)[0].max(dim=1, keepdim=True)[0]  # [B, 1]
    rope_deltas = (maxpos + 1 - new_inputs_embeds.shape[1]).to(dtype=torch.long, device=dev)
    cache_position = torch.arange(new_inputs_embeds.shape[1], device=dev, dtype=torch.int32)

    return new_inputs_embeds, new_position_ids, new_attention_mask, rope_deltas, cache_position, new_input_ids, new_labels
