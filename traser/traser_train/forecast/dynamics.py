"""Object-centric latent dynamics: history window latents -> future window latents + boxes.

For every object TRASER already produces one group of ``R`` latents per observed temporal
window (TWR) plus a whole-trajectory summary (OTR). The forecaster predicts the TWR latents
of the next ``F`` windows directly (horizon-conditioned, not autoregressive, so errors do not
compound over the horizon):

1. every observed (object, window) becomes one token: pooled latents + its box + time and
   object embeddings; the OTR summary adds one token per object;
2. one query token per (object, future window);
3. a transformer encoder runs over all tokens of the clip, so objects see each other (hand
   approaching a bottle) and their own past;
4. each future query is expanded back to ``R`` latents by a small decoder that attends to the
   object's last observed latents, and added as a residual to those latents.

``out_proj`` and the box ``delta`` layer start at zero, so before training the forecaster
is exactly the copy-last baseline (future latents = last observed latents, future box = last
observed box). Training can only move it away from that baseline where that helps.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectLatentForecaster(nn.Module):
    def __init__(self, hidden_size: int, n_latents: int, d_model: int = 512, n_layers: int = 4,
                 n_decoder_layers: int = 2, n_heads: int = 8, max_windows: int = 32,
                 max_objects: int = 64, box_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size, self.n_latents, self.d_model, self.box_dim = hidden_size, n_latents, d_model, box_dim
        self.max_windows, self.max_objects = max_windows, max_objects

        self.in_norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, d_model)
        self.box_in = nn.Linear(box_dim + 1, d_model)
        self.time_emb = nn.Embedding(max_windows, d_model)
        self.kind_emb = nn.Embedding(3, d_model)          # 0 observed window, 1 future query, 2 OTR summary
        self.obj_emb = nn.Embedding(max_objects, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers, enable_nested_tensor=False)
        self.latent_queries = nn.Embedding(n_latents, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, hidden_size)

        self.box_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        self.box_delta = nn.Linear(d_model, box_dim)
        self.vis_logit = nn.Linear(d_model, 1)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Explicit init (HF ``_init_weights`` overwrites new modules when loading a checkpoint)."""
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()  # its in_proj_weight is a bare Parameter, not an nn.Linear
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Start as the copy-last baseline.
        for m in (self.out_proj, self.box_delta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.vis_logit.bias, 2.0)  # objects usually stay visible

    @staticmethod
    def last_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (O, W, ...), valid: (O, W) -> x at each object's last valid window (O, ...)."""
        W = valid.shape[1]
        idx = torch.where(valid, torch.arange(W, device=valid.device)[None], torch.full_like(valid, -1, dtype=torch.long))
        last = idx.max(dim=1).values.clamp(min=0)
        return x[torch.arange(x.shape[0], device=x.device), last]

    def forward(self, hist_latents: torch.Tensor, hist_valid: torch.Tensor, n_future: int,
                otr_latents: Optional[torch.Tensor] = None,
                hist_boxes: Optional[torch.Tensor] = None,
                hist_box_valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        hist_latents:   (O, Wh, R, D)  TWR latents of the observed windows (zeros where empty)
        hist_valid:     (O, Wh) bool   window has TWR latents
        otr_latents:    (O, R2, D)     OTR summary, optional
        hist_boxes:     (O, Wh, box_dim), hist_box_valid: (O, Wh)
        Returns future_latents (O, F, R, D), future_boxes (O, F, box_dim), vis_logits (O, F),
        copy_last_latents (O, R, D) and copy_last_boxes (O, box_dim).
        """
        O, Wh, R, D = hist_latents.shape
        dev = hist_latents.device
        if O > self.max_objects or Wh + n_future > self.max_windows:
            raise ValueError(f"{O} objects / {Wh + n_future} windows exceed the forecaster limits "
                             f"({self.max_objects} / {self.max_windows})")
        dtype = self.in_proj.weight.dtype
        hist_tok = self.in_proj(self.in_norm(hist_latents.to(dtype))).mean(dim=2)    # (O, Wh, d)
        if hist_boxes is None:
            hist_boxes = torch.zeros(O, Wh, self.box_dim, device=dev)
            hist_box_valid = torch.zeros(O, Wh, dtype=torch.bool, device=dev)
        box_feat = torch.cat([hist_boxes.to(dtype) * hist_box_valid[..., None],
                              hist_box_valid[..., None].to(dtype)], dim=-1)
        obj_ids = torch.arange(O, device=dev)
        hist_tok = (hist_tok + self.box_in(box_feat) + self.time_emb(torch.arange(Wh, device=dev))[None]
                    + self.kind_emb.weight[0] + self.obj_emb(obj_ids)[:, None])
        hist_ok = hist_valid | hist_box_valid

        fut_tok = (self.time_emb(torch.arange(Wh, Wh + n_future, device=dev))[None]
                   + self.kind_emb.weight[1] + self.obj_emb(obj_ids)[:, None]).expand(O, n_future, -1)

        tokens = [hist_tok, fut_tok]
        masks = [hist_ok, torch.ones(O, n_future, dtype=torch.bool, device=dev)]
        if otr_latents is not None:
            otr_tok = self.in_proj(self.in_norm(otr_latents.to(dtype))).mean(dim=1)
            otr_tok = otr_tok + self.kind_emb.weight[2] + self.obj_emb(obj_ids)
            tokens.append(otr_tok[:, None])
            masks.append(torch.ones(O, 1, dtype=torch.bool, device=dev))
        seq = torch.cat(tokens, dim=1).reshape(1, -1, self.d_model)
        keep = torch.cat(masks, dim=1).reshape(1, -1)
        enc = self.encoder(seq, src_key_padding_mask=~keep).reshape(O, -1, self.d_model)
        z = enc[:, Wh:Wh + n_future]                                                 # (O, F, d)

        # Expand each future query back to R latents, as a residual on the last observed latents.
        copy_last = self.last_valid(hist_latents, hist_valid)                         # (O, R, D)
        memory = self.in_proj(self.in_norm(copy_last.to(dtype)))                      # (O, R, d)
        memory = memory[:, None].expand(O, n_future, R, self.d_model).reshape(O * n_future, R, self.d_model)
        tgt = self.latent_queries.weight[None] + z.reshape(O * n_future, 1, self.d_model)
        dec = self.decoder(tgt, memory)
        delta = self.out_proj(self.out_norm(dec)).reshape(O, n_future, R, D)
        future_latents = copy_last[:, None].to(delta.dtype) + delta

        last_box = self.last_valid(hist_boxes, hist_box_valid)                         # (O, box_dim)
        h = self.box_mlp(torch.cat([z, self.box_in(torch.cat(
            [last_box.to(dtype), torch.ones(O, 1, device=dev, dtype=dtype)], -1))[:, None].expand_as(z)], -1))
        future_boxes = (last_box[:, None].float() + self.box_delta(h).float()).clamp(0.0, 1.0)
        vis_logits = self.vis_logit(h).squeeze(-1).float()
        return dict(future_latents=future_latents, future_boxes=future_boxes, vis_logits=vis_logits,
                    copy_last_latents=copy_last, copy_last_boxes=last_box)


# --------------------------------------------------------------------------- losses
def latent_distillation_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """MSE between layer-normalized predicted and teacher latents on the valid windows.

    pred/target: (O, F, R, D), valid: (O, F). Normalizing makes the loss scale-free (the
    resampler output scale drifts while it is trained).
    """
    if valid.sum() == 0:
        return pred.sum() * 0.0
    p = F.layer_norm(pred.float(), pred.shape[-1:])
    t = F.layer_norm(target.float(), target.shape[-1:])
    err = ((p - t) ** 2).mean(dim=(-1, -2))                                          # (O, F)
    return (err * valid).sum() / valid.sum()


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise GIoU of xyxy boxes (..., 4)."""
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-6)
    lt_c = torch.min(a[..., :2], b[..., :2])
    rb_c = torch.max(a[..., 2:], b[..., 2:])
    area_c = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def box_losses(pred_boxes, vis_logits, gt_boxes, gt_valid, l1_weight=5.0, giou_weight=2.0, vis_weight=1.0):
    """L1 + GIoU on the windows where the object is visible, BCE on visibility (DETR weights)."""
    vis = F.binary_cross_entropy_with_logits(vis_logits, gt_valid.float())
    if gt_valid.sum() == 0:
        zero = pred_boxes.sum() * 0.0
        return dict(box_l1=zero, box_giou=zero, vis=vis, total=vis_weight * vis)
    p, g = pred_boxes[gt_valid], gt_boxes[gt_valid].to(pred_boxes.dtype)
    l1 = (p - g).abs().sum(-1).mean()
    giou = (1 - generalized_iou(box_cxcywh_to_xyxy(p), box_cxcywh_to_xyxy(g))).mean()
    return dict(box_l1=l1, box_giou=giou, vis=vis, total=l1_weight * l1 + giou_weight * giou + vis_weight * vis)
