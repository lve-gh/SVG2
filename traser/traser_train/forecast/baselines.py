"""Training-free forecasting baselines, scored on exactly the clips the model is scored on.

Relationships (they see the ground-truth observed graph, i.e. more than the model gets):
    copy      relations active in the last observed second persist over the horizon
    markov    per-pair predicate Markov chain fitted on the training split, k-step argmax
Boxes:
    copy      last observed box
    velocity  constant velocity from the last two observed windows

    python -m traser_train.forecast.baselines --train data/psg4d_hoi_train.json \
        --test data/psg4d_hoi_test.json --history_sec 3 --horizon_sec 3
"""

import argparse
import json
from collections import Counter, defaultdict

import torch

from .data_forecast import ClipSpec, clip_targets, load_annotations, relationships_in_seconds
from .metrics import ForecastMeter, normalize_predicate

NONE = "<none>"


def pair_states(rels, n_sec):
    """{(s, o): [set of predicates active at second t] for t < n_sec} from second-level rels."""
    states = defaultdict(lambda: [set() for _ in range(n_sec)])
    for s, p, o, spans in rels:
        for a, b in spans:
            for t in range(max(a, 0), min(b, n_sec - 1) + 1):
                states[(s, o)][t].add(normalize_predicate(p))
    return states


class MarkovBaseline:
    """P(predicate at t+1 | predicate at t) per (subject class, object class), with a global backoff."""

    def __init__(self, min_count: int = 20):
        self.min_count = min_count
        self.counts = defaultdict(lambda: defaultdict(Counter))   # (cls_s, cls_o) -> a -> Counter(b)
        self.global_counts = defaultdict(Counter)
        self.pairs = set()

    def fit(self, items):
        for it in items:
            n_sec = int(it["num_frames"] / it["fps"])
            names = {o["id"]: o["name"] for o in it["objects"]}
            ident = {o["id"]: o["id"] for o in it["objects"]}
            rels = relationships_in_seconds(it.get("relationships", []), float(it["fps"]), 0, 0, n_sec, ident)
            for (s, o), seq in pair_states(rels, n_sec).items():
                key = (names.get(s), names.get(o))
                self.pairs.add(key)
                for t in range(n_sec - 1):
                    for a in (seq[t] or {NONE}):
                        for b in (seq[t + 1] or {NONE}):
                            self.counts[key][a][b] += 1
                            self.global_counts[a][b] += 1
        return self

    def _row(self, key, a):
        c = self.counts[key][a]
        return c if sum(c.values()) >= self.min_count else self.global_counts[a]

    def predict(self, observed, names, history, horizon):
        """observed: second-level rels of the clip in "Object k" numbering; names[k-1] = class."""
        preds = []
        states = pair_states(observed, history)
        K = len(names)
        for s in range(1, K + 1):
            for o in range(1, K + 1):
                key = (names[s - 1], names[o - 1])
                if s == o or key not in self.pairs:
                    continue
                last = states[(s, o)][history - 1] if (s, o) in states else set()
                dist = Counter({p: 1.0 for p in (last or {NONE})})
                for t in range(history, history + horizon):
                    nxt = Counter()
                    total = sum(dist.values())
                    for a, w in dist.items():
                        row = self._row(key, a)
                        n = sum(row.values())
                        for b, c in row.items():
                            nxt[b] += w / total * c / max(n, 1)
                    dist = nxt if nxt else dist
                    best = max(dist, key=dist.get) if dist else NONE
                    if best != NONE:
                        preds.append([s, best, o, [[t, t]]])
        return preds


def copy_last_relations(observed, history, horizon):
    out = []
    for s, p, o, spans in observed:
        if any(a <= history - 1 <= b for a, b in spans):
            out.append([s, p, o, [[history, history + horizon - 1]]])
    return out


def box_baseline(boxes, valid, n_hist, kind):
    """boxes/valid: (O, W, 4)/(O, W) over the whole clip -> forecast (O, F, 4)."""
    O, W, _ = boxes.shape
    Fw = W - n_hist
    out = torch.zeros(O, Fw, 4)
    for o in range(O):
        idx = [w for w in range(n_hist) if valid[o, w]]
        last = boxes[o, idx[-1]] if idx else torch.tensor([0.5, 0.5, 0.0, 0.0])
        vel = torch.zeros(4)
        if kind == "velocity" and len(idx) >= 2:
            vel = (boxes[o, idx[-1]] - boxes[o, idx[-2]]) / (idx[-1] - idx[-2])
            vel[2:] = 0  # keep the size, move the centre
        for f in range(Fw):
            step = n_hist + f - (idx[-1] if idx else n_hist - 1)
            out[o, f] = (last + vel * step).clamp(0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--history_sec", type=int, default=3)
    ap.add_argument("--horizon_sec", type=int, default=3)
    ap.add_argument("--eval_stride", type=int, default=1)
    ap.add_argument("--sample_fps", type=int, default=2)
    ap.add_argument("--window_sec", type=int, default=1)
    ap.add_argument("--object_filter", default="related")
    ap.add_argument("--max_objects", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    markov = MarkovBaseline().fit(load_annotations(args.train))
    meters = {name: ForecastMeter() for name in ("copy+copy", "markov+velocity")}
    H, Fh = args.history_sec, args.horizon_sec
    for item in load_annotations(args.test):
        mask_data = json.load(open(item["mask_json"]))
        dur = int(item["num_frames"] / item["fps"])
        for t in range(H, dur - Fh + 1, args.eval_stride):
            ct = clip_targets(item, ClipSpec(0, t, H, Fh), mask_data, args)
            if not ct["obj_ids"]:
                continue
            nh = ct["n_hist_windows"]
            gt_b, gt_v = ct["boxes"][:, nh:], ct["box_valid"][:, nh:]
            meters["copy+copy"].add(copy_last_relations(ct["observed"], H, Fh), ct["target"], H, Fh,
                                    box_baseline(ct["boxes"], ct["box_valid"], nh, "copy"), gt_b, gt_v)
            meters["markov+velocity"].add(markov.predict(ct["observed"], ct["obj_names"], H, Fh), ct["target"],
                                          H, Fh, box_baseline(ct["boxes"], ct["box_valid"], nh, "velocity"),
                                          gt_b, gt_v)
    results = {k: m.summary() for k, m in meters.items()}
    print(json.dumps(results, indent=2))
    if args.output:
        json.dump(results, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
