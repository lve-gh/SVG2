"""Trainer for TRASER-F: LM loss on the future relationships + latent distillation + boxes."""

from collections import defaultdict

import torch
from transformers import Trainer

from ..train.trainer_insert import MAX_ARRANGED_SEQ_LEN
from .dynamics import box_losses, latent_distillation_loss
from .step import build_forecast_inputs


class ForecastTrainer(Trainer):
    _cpu_only_keys = ("obj_masks", "obj_labels", "n_hist_windows", "n_fut_windows", "hist_boxes",
                      "hist_box_valid", "fut_boxes", "fut_box_valid", "meta")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_sums = defaultdict(float)
        self._aux_count = 0

    def create_optimizer(self):
        """Parameter groups with their own learning rates: forecaster, resamplers, merger, rest."""
        if self.optimizer is not None:
            return self.optimizer
        a = self.args
        groups = {"forecaster": (a.forecaster_lr, 0.0), "resampler": (a.resampler_lr, 0.0),
                  "merger": (a.mm_projector_lr, a.weight_decay)}
        buckets = {k: [] for k in groups}
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            key = next((k for k in groups if k in n and groups[k][0]), None)
            if key is not None:
                buckets[key].append(p)
            elif p.ndim < 2 or "bias" in n or "norm" in n or "embed_tokens" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [{"params": decay, "weight_decay": a.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0}]
        param_groups += [{"params": buckets[k], "lr": lr, "weight_decay": wd}
                         for k, (lr, wd) in groups.items() if buckets[k]]
        param_groups = [g for g in param_groups if g["params"]]
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(a)
        self.optimizer = cls(param_groups, **kwargs)
        return self.optimizer

    def _prepare_inputs(self, inputs):
        cpu_side = {k: inputs.pop(k) for k in list(inputs) if k in self._cpu_only_keys}
        prepared = super()._prepare_inputs(inputs)
        prepared.update(cpu_side)
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        a = self.args
        base = self.accelerator.unwrap_model(model)
        tok = self.processing_class
        emb, pid, mask, rope, cache, labels, aux = build_forecast_inputs(
            base, tok, inputs, sample_fps=a.sample_fps, window_sec=a.window_sec,
            coverage_thresh=a.coverage_thresh, time_reduce=a.time_reduce, mode=a.forecast_mode,
            with_teacher=(a.lambda_latent > 0 or a.forecast_mode == "oracle"))
        if emb.shape[1] > MAX_ARRANGED_SEQ_LEN:
            # Truncating would cut the answer; skip the LM term for this batch instead.
            print(f"[WARNING] arranged sequence {emb.shape[1]} > {MAX_ARRANGED_SEQ_LEN}: {[m['video'] for m in inputs['meta']]}")
            emb, pid, mask, labels = emb[:, :MAX_ARRANGED_SEQ_LEN], pid[..., :MAX_ARRANGED_SEQ_LEN], \
                mask[:, :MAX_ARRANGED_SEQ_LEN], torch.full_like(labels[:, :MAX_ARRANGED_SEQ_LEN], -100)
            cache = cache[:MAX_ARRANGED_SEQ_LEN]

        outputs = model(inputs_embeds=emb, position_ids=pid, attention_mask=mask, rope_deltas=rope,
                        cache_position=cache, labels=labels, use_cache=False)
        lm = outputs.loss if outputs.loss is not None and torch.isfinite(outputs.loss) else emb.sum() * 0.0
        loss = lm

        lat_terms, box_terms = [], []
        for b, out in enumerate(aux):
            if a.lambda_latent > 0 and a.forecast_mode == "latent":
                lat_terms.append(latent_distillation_loss(out["future_latents"], out["teacher_latents"],
                                                          out["teacher_valid"]))
            if a.lambda_box > 0:
                box_terms.append(box_losses(out["future_boxes"], out["vis_logits"],
                                            inputs["fut_boxes"][b].to(emb.device),
                                            inputs["fut_box_valid"][b].to(emb.device)))
        logs = {"lm_loss": lm.detach().float().item()}
        if lat_terms:
            lat = torch.stack(lat_terms).mean()
            loss = loss + a.lambda_latent * lat
            logs["latent_loss"] = lat.detach().float().item()
        if box_terms:
            box = torch.stack([t["total"] for t in box_terms]).mean()
            loss = loss + a.lambda_box * box
            for k in ("box_l1", "box_giou", "vis"):
                logs[k] = torch.stack([t[k] for t in box_terms]).mean().detach().float().item()
        for k, v in logs.items():
            self._aux_sums[k] += v
        self._aux_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._aux_count and "loss" in logs:
            for k, v in self._aux_sums.items():
                logs[k] = round(v / self._aux_count, 4)
            self._aux_sums.clear()
            self._aux_count = 0
        super().log(logs, *args, **kwargs)
