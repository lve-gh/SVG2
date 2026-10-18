"""Forecasting metrics.

A relationship is scored per future second: the unit is the quadruple
(subject, predicate, object, second). For every clip and horizon step k = 1..F:

* R / mR      recall of ground-truth quadruples (micro / averaged over predicates);
* P / F1      precision, so that a model cannot buy recall by listing every predicate;
* R@k         recall restricted to the k-th second after the observation;
* box mIoU@k  IoU of the forecast box vs. the real box (objects visible at that second);
* ADE / FDE   L2 error of the box centre (fraction of the frame), mean over the horizon / last step;
* R_loc       SGF-style recall: a quadruple counts only if both forecast boxes have IoU >= 0.5.

Generated text is unranked, so recall is reported over all predicted quadruples rather than
at a fixed K. Methods that rank (ImparTail, SceneSayer) are scored with ``recall_at_k``.
"""

import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

Quad = Tuple[int, str, int, int]


def normalize_predicate(p: str) -> str:
    return re.sub(r"\s+", " ", str(p).strip().lower())


def parse_relationships(text: str) -> List[list]:
    """Relationships from generated text; tolerant to truncation and extra prose."""
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            rels = data.get("relationships", []) if isinstance(data, dict) else []
            return [r for r in rels if isinstance(r, list) and len(r) >= 4]
        except json.JSONDecodeError:
            pass
    out = []
    for m in re.finditer(r'\[\s*(-?\d+)\s*,\s*"([^"]+)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[\d\s,\[\]]*\]\s*\])', text):
        try:
            out.append([int(m.group(1)), m.group(2), int(m.group(3)), json.loads(m.group(4))])
        except json.JSONDecodeError:
            continue
    return out


def to_quads(rels: Iterable[list], lo: int, hi: int) -> Set[Quad]:
    """Quadruples of ``rels`` within seconds ``[lo, hi)``."""
    quads = set()
    for r in rels:
        try:
            s, p, o, spans = int(r[0]), normalize_predicate(r[1]), int(r[2]), r[3]
            for a, b in spans:
                for t in range(max(int(a), lo), min(int(b), hi - 1) + 1):
                    quads.add((s, p, o, t))
        except (TypeError, ValueError):
            continue
    return quads


def box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def xyxy(x):
        return torch.stack([x[..., 0] - x[..., 2] / 2, x[..., 1] - x[..., 3] / 2,
                            x[..., 0] + x[..., 2] / 2, x[..., 1] + x[..., 3] / 2], -1)
    a, b = xyxy(a.float()), xyxy(b.float())
    lt, rb = torch.max(a[..., :2], b[..., :2]), torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).prod(-1) + (b[..., 2:] - b[..., :2]).prod(-1) - inter
    return inter / union.clamp(min=1e-6)


class ForecastMeter:
    def __init__(self, max_horizon: int = 5):
        self.max_horizon = max_horizon
        self.tp = self.n_gt = self.n_pred = self.tp_loc = 0
        self.tp_k = defaultdict(int)
        self.gt_k = defaultdict(int)
        self.pred_tp = defaultdict(int)
        self.pred_gt = defaultdict(int)
        self.iou_k = defaultdict(list)
        self.center_err_k = defaultdict(list)
        self.vis_correct = self.vis_total = 0
        self.n_clips = 0

    def add(self, pred_rels: List[list], gt_rels: List[list], history: int, horizon: int,
            pred_boxes: Optional[torch.Tensor] = None, gt_boxes: Optional[torch.Tensor] = None,
            gt_valid: Optional[torch.Tensor] = None, vis_logits: Optional[torch.Tensor] = None,
            window_sec: int = 1):
        """Boxes are (O, F, 4) normalized cx, cy, w, h; row k is "Object k+1"."""
        self.n_clips += 1
        lo, hi = history, history + horizon
        P, G = to_quads(pred_rels, lo, hi), to_quads(gt_rels, lo, hi)
        hit = P & G
        self.tp += len(hit)
        self.n_gt += len(G)
        self.n_pred += len(P)
        for q in G:
            k = q[3] - history + 1
            self.gt_k[k] += 1
            self.pred_gt[q[1]] += 1
            if q in hit:
                self.tp_k[k] += 1
                self.pred_tp[q[1]] += 1

        if pred_boxes is None or gt_boxes is None:
            return
        iou = box_iou_cxcywh(pred_boxes, gt_boxes)                                  # (O, F)
        err = (pred_boxes[..., :2].float() - gt_boxes[..., :2].float()).norm(dim=-1)
        for f in range(gt_valid.shape[1]):
            for o in range(gt_valid.shape[0]):
                if gt_valid[o, f]:
                    self.iou_k[f + 1].append(float(iou[o, f]))
                    self.center_err_k[f + 1].append(float(err[o, f]))
        if vis_logits is not None:
            self.vis_correct += int(((vis_logits > 0) == gt_valid.bool()).sum())
            self.vis_total += gt_valid.numel()
        for s, _, o, t in hit:
            f = (t - history) // window_sec
            ok = all(1 <= x <= iou.shape[0] and gt_valid[x - 1, f] and iou[x - 1, f] >= 0.5 for x in (s, o))
            self.tp_loc += int(ok)

    def summary(self) -> Dict[str, float]:
        def div(a, b):
            return round(100.0 * a / b, 2) if b else float("nan")
        r, p = div(self.tp, self.n_gt), div(self.tp, self.n_pred)
        out = {"clips": self.n_clips, "R": r, "P": p,
               "F1": round(2 * r * p / (r + p), 2) if r == r and p == p and (r + p) > 0 else float("nan"),
               "mR": round(sum(100.0 * self.pred_tp[k] / v for k, v in self.pred_gt.items()) / len(self.pred_gt), 2)
               if self.pred_gt else float("nan")}
        for k in sorted(self.gt_k):
            out[f"R@{k}s"] = div(self.tp_k[k], self.gt_k[k])
        if self.iou_k:
            out["R_loc"] = div(self.tp_loc, self.n_gt)
            for k in sorted(self.iou_k):
                out[f"mIoU@{k}s"] = round(100.0 * sum(self.iou_k[k]) / len(self.iou_k[k]), 2)
            ks = sorted(self.center_err_k)
            all_err = [e for k in ks for e in self.center_err_k[k]]
            out["ADE"] = round(sum(all_err) / len(all_err), 4)
            out["FDE"] = round(sum(self.center_err_k[ks[-1]]) / len(self.center_err_k[ks[-1]]), 4)
        if self.vis_total:
            out["vis_acc"] = div(self.vis_correct, self.vis_total)
        return out


def recall_at_k(scored: List[Tuple[Quad, float]], gt: Set[Quad], k: int) -> float:
    """R@K for ranked predictions (one clip)."""
    top = {q for q, _ in sorted(scored, key=lambda x: -x[1])[:k]}
    return len(top & gt) / len(gt) if gt else float("nan")
