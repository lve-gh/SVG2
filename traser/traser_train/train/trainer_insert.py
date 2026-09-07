import math

import torch
from transformers import Trainer

from .token_arrangement import rearrange_token
from .token_selection import select_tokens

# Qwen2.5-VL vision constants.
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 14

# Hard cap on the rearranged sequence length; longer sequences are truncated to
# keep peak GPU memory bounded.
MAX_ARRANGED_SEQ_LEN = 20000

OBJECT_LABEL_TEMPLATE = "Object {i}: "


class TraserTrainer(Trainer):
    """Trainer that turns each batch into trajectory-aligned object blocks.

    Per step: select each object's visual tokens from its segmentation masks
    (``select_tokens``), rebuild the input sequence with per-object OTR/TWR latents
    (``rearrange_token``), then run the model on the rearranged embeddings.
    """

    # Keys that must stay on CPU (used only for planning, not by the model forward).
    _cpu_only_keys = ("obj_masks", "video_name")

    def _get_tok(self):
        try:
            proc = self.processing_class
        except AttributeError:
            proc = getattr(self, "tokenizer", None)
        if proc is None:
            raise RuntimeError("No processing_class/tokenizer available on Trainer.")
        return proc

    def _prepare_inputs(self, inputs):
        # Keep CPU-only fields out of the default device placement.
        cpu_side = {}
        for k in list(inputs.keys()):
            if k in self._cpu_only_keys and inputs[k] is not None:
                cpu_side[k] = inputs.pop(k)

        prepared = super()._prepare_inputs(inputs)
        prepared.update(cpu_side)
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            base = self.accelerator.unwrap_model(model)
        except Exception:
            from accelerate.utils import extract_model_from_parallel
            base = extract_model_from_parallel(model)

        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        inputs.pop("pixel_values", None)
        inputs.pop("image_grid_thw", None)
        pixel_values_videos = inputs.pop("pixel_values_videos")
        video_grid_thw = inputs.pop("video_grid_thw")
        second_per_grid_ts = inputs.pop("second_per_grid_ts", None)
        video_names = inputs.pop("video_name", None)
        labels = inputs.pop("labels", None)
        obj_masks_list = inputs.pop("obj_masks")

        coverage_thresh = getattr(self.args, "coverage_thresh", 0.5)
        time_reduce = getattr(self.args, "time_reduce", "max")
        training_fps = getattr(self.args, "training_fps", 1)
        temporal_window_length = getattr(self.args, "temporal_window_length", 4)

        # ---- (1) Token selection: mask coverage -> per-object video-token indices ----
        per_obj_idx_batch = []
        num_grids_batch = []
        for b, obj_masks in enumerate(obj_masks_list):
            # video_grid_thw is the post-merge grid (T, H, W).
            T, Hm, Wm = video_grid_thw[b] if video_grid_thw[b].shape[0] == 3 else video_grid_thw[b][0]
            T = T.to("cpu")
            Hm = Hm.to("cpu")
            Wm = Wm.to("cpu")
            num_grids_batch.append(int(T.item()))

            with torch.no_grad():
                # Run the selection on CPU to avoid GPU memory pressure.
                _, per_obj_idx, _ = select_tokens(
                    obj_masks=obj_masks.to("cpu"),
                    grid_thw=[T, Hm, Wm],
                    patch_size=PATCH_SIZE,
                    spatial_merge_size=SPATIAL_MERGE_SIZE,
                    temporal_patch_size=TEMPORAL_PATCH_SIZE,
                    coverage_thresh=coverage_thresh,
                    time_reduce=time_reduce,
                    device="cpu",
                )
                per_obj_idx = [obj_idx.to(input_ids.device) for obj_idx in per_obj_idx]
                per_obj_idx_batch.append(per_obj_idx)

                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        del obj_masks_list
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        tok = self._get_tok()

        # ---- (2) "Object k: " label tokens for each object ----
        text_token_ids_per_sample = []
        for per_obj_idx in per_obj_idx_batch:
            texts = [OBJECT_LABEL_TEMPLATE.format(i=(k + 1)) for k in range(len(per_obj_idx))]
            if len(texts) == 0:
                text_token_ids_per_sample.append([])
                continue
            enc = tok(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            text_token_ids_per_sample.append([torch.tensor(x, dtype=torch.long) for x in enc])

        # ---- (3) Timestamp tokens "<s - e sec>" for each temporal window ----
        # One temporal grid spans TEMPORAL_PATCH_SIZE frames = TEMPORAL_PATCH_SIZE / fps seconds.
        seconds_per_grid = TEMPORAL_PATCH_SIZE / training_fps
        grids_per_temporal_window = int(temporal_window_length / seconds_per_grid)
        timestamp_token_ids_per_batch = []
        grids_per_temporal_window_per_batch = []
        for num_grids in num_grids_batch:
            temporal_window_num = math.ceil(num_grids / grids_per_temporal_window)
            temporal_text_list = []
            for w_id in range(temporal_window_num):
                start_time = w_id * temporal_window_length
                end_time = start_time + temporal_window_length
                temporal_text_list.append(f"<{int(start_time)} - {int(end_time)} sec>")
            enc = tok(
                temporal_text_list,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            timestamp_token_ids_per_batch.append([torch.tensor(x) for x in enc])
            grids_per_temporal_window_per_batch.append(grids_per_temporal_window)

        # ---- (4) Rearrange the sequence into trajectory-aligned object blocks ----
        new_emb, new_pid, new_mask, rope_deltas, cache_pos, new_input_ids, new_labels = rearrange_token(
            model=base,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            image_grid_thw=None,
            pixel_values=None,
            second_per_grid_ts=second_per_grid_ts,
            obj_token_indices_per_sample=per_obj_idx_batch,
            obj_traj_start_id=getattr(self.args, "obj_traj_start_id", None),
            obj_traj_end_id=getattr(self.args, "obj_traj_end_id", None),
            use_resampler=getattr(self.args, "use_resampler", True),
            use_second_resampler=getattr(self.args, "object_resampler", True),
            text_token_ids_per_sample=text_token_ids_per_sample,
            timestamp_token_ids_per_batch=timestamp_token_ids_per_batch,
            grids_per_temporal_window_per_batch=grids_per_temporal_window_per_batch,
            labels=labels,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- (5) Truncate overly long sequences to bound peak memory ----
        if new_emb.size(1) > MAX_ARRANGED_SEQ_LEN:
            print(
                f"[WARNING] Truncating {video_names} from {new_emb.size(1)} to {MAX_ARRANGED_SEQ_LEN} tokens"
            )
            new_emb = new_emb[:, :MAX_ARRANGED_SEQ_LEN, :]
            new_mask = new_mask[:, :MAX_ARRANGED_SEQ_LEN]
            new_pid = new_pid[:, :, :MAX_ARRANGED_SEQ_LEN]  # [3, B, L]
            if new_labels is not None:
                new_labels = new_labels[:, :MAX_ARRANGED_SEQ_LEN]

            maxpos = new_pid.max(dim=0)[0].max(dim=1, keepdim=True)[0]  # [B, 1]
            rope_deltas = (maxpos + 1 - new_emb.size(1)).to(dtype=torch.long, device=new_emb.device)
            cache_pos = torch.arange(new_emb.size(1), dtype=torch.int32, device=new_emb.device)

        outputs = model(
            inputs_embeds=new_emb,
            position_ids=new_pid,
            attention_mask=new_mask,
            rope_deltas=rope_deltas,
            cache_position=cache_pos,
            labels=new_labels,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
