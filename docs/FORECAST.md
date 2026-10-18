# TRASER-F: scene graph forecasting

Given the first **H** seconds of a clip and the mask trajectories of its objects, predict the
relationship triplets of the next **F** seconds (H, F ∈ 1–5 s) **and** where every object will
be. Code: [`traser/traser_train/forecast/`](../traser/traser_train/forecast).

## Task

| | |
|---|---|
| Input | RGB frames of `[T-H, T)` at 2 fps, per-object masks on those frames, object categories |
| Output | `{"relationships": [[s, pred, o, [[t0, t1]]], ...]}` for seconds `H … H+F-1` of the clip, and one box (cx, cy, w, h) + visibility per object per future second |
| Objects | those visible in the observed part; the future introduces none (as in SGA) |

This matches the GAGS / PGAGS settings of scene graph anticipation (objects given, relations
anticipated) and adds the localization part of scene graph *forecasting* (FDSG).

## Related work, and what we take from it

| Method | Venue | How it forecasts | Boxes | Takeaway for TRASER-F |
|---|---|---|---|---|
| STTran+/++, DSG-DETR+/++ | baselines of SceneSayer | anticipatory transformer, autoregressive | no | autoregressive rollout compounds errors |
| SceneSayer ODE/SDE | ECCV 2024 | latent dynamics of pair features (NeuralODE/SDE) | regressed | **forecast in latent space**, decode with the same heads |
| ImparTail | CVPR 2025 | SceneSayer/STTran trained with a masking curriculum on partial labels | — | **robust to missing labels**: relevant for auto-labelled SVG2 data |
| FDSG | arXiv 2506.01487 | NeuralSDE on DETR query features + identity-plus-residual box dynamics | **yes** | **boxes = last box + residual**; defines SGF metrics (IoU ≥ 0.5) |
| HyperGLM | CVPR 2025 | hypergraph + multimodal LLM | no | an LLM decoder is competitive on SGA |
| OOTSM | arXiv 2509.05661 | LLM on textual scene graphs, objects first then relations | no | LLM forecasting gains most in mR (+21.9 mR@50 vs SceneSayerSDE) |

None of them works on per-object trajectory tokens of a video LLM. TRASER-F puts the
SceneSayer/FDSG idea (latent dynamics, horizon-conditioned, residual on the last state) inside
TRASER's own representation and keeps TRASER's LLM as the decoder (HyperGLM/OOTSM).

## Architecture

```
history frames ─► Qwen2.5-VL ViT ─► mask-based token selection ─► per object:
                                                                   OTR latents (whole history)
                                                                   TWR latents per 1 s window
                                         ┌─────────────────────────────────┘
                                         ▼
                     Object latent forecaster (new, ~24M params)
        tokens: one per (object, observed window) + OTR + one query per (object, future window)
        transformer encoder over all objects  →  decoder expands each query to R latents
        future latents = last observed latents + Δ        future box = last box + Δ (+ visibility)
                                         │
                                         ▼
  <obj_traj_start> Object k (bottle): <|vision_start|> [OTR]
      <0 - 1 sec> [TWR] … <H-1 - H sec> [TWR]
      <forecast H - H+1 sec> [future latents] … <forecast H+F-1 - H+F sec> [future latents]
  <|vision_end|> <obj_traj_end>      ×  every object
                                         │
                                         ▼
               Qwen2.5-VL LLM (TRASER weights) ─► future relationships as JSON
```

Design choices:

* **Forecast in TRASER's latent space.** The future windows are filled with latents of the
  same kind the LLM already reads, so the scene graph decoder of TRASER is reused as-is.
* **Horizon-conditioned, not autoregressive.** Every future window is predicted directly from
  the history (query = object + absolute time), so a 5 s horizon does not compound errors.
* **Objects attend to each other.** All (object, window) tokens of a clip share one encoder:
  the hand approaching the bottle changes the bottle's future.
* **Starts as the copy-last baseline.** The output layers are zero-initialized: before
  training, future latents = last observed latents and future box = last box. Training only
  moves away from persistence where that helps.
* **Free dense supervision.** During training the real future frames are available, so the
  resampler's latents of the future windows are distillation targets (no labels needed).
* **No leakage.** Qwen2.5-VL's vision encoder attends within one temporal grid, so encoding
  the whole clip gives the same history features as encoding the history alone (checked by the
  smoke test); evaluation feeds only the observed frames anyway.

Loss: `L = L_LM(future JSON) + λ_lat · MSE(LN(pred), LN(teacher)) + λ_box · (5·L1 + 2·GIoU + BCE_vis)`.

Training (`traser/scripts/train_forecast.sh`):

| Stage | Trainable | LR | Purpose |
|---|---|---|---|
| A | forecaster only | 1e-4 | learn dynamics against a frozen TRASER (cheap, stable teacher) |
| B | LLM + merger + resamplers + forecaster | 1e-5 / 2e-5 / 5e-5 / 5e-5 | adapt the decoder to forecasting |

Clips are drawn on the fly: random H, F ∈ [1, 5], anchors biased (70 %) towards a relation
starting in the future window, empty futures kept 10 % of the time.

## Hypotheses and ablations

All run with the same data, schedule and evaluation; only the flag changes.

| ID | Hypothesis | Run |
|---|---|---|
| B0 | Persistence is a strong baseline | `baselines.py` copy / Markov, `--mode copy` |
| H1 | The LLM can forecast from history blocks alone | `MODE=prompt` (no future slots) |
| H2 | Latent dynamics beat persistence (main model) | `MODE=latent` |
| H2a | Latent distillation is what makes H2 work | `MODE=latent LAMBDA_LATENT=0` |
| H2b | Boxes help the relations (shared dynamics) | `MODE=latent LAMBDA_BOX=0` |
| UB | Upper bound if dynamics were perfect | `evaluate.py --mode oracle` (real future latents) |
| H3 | History length matters | evaluate one model at H = 1 … 5 |

`oracle` vs `latent` vs `copy` splits the error into "dynamics" and "decoding": if `oracle` is
not much better than `copy`, the bottleneck is the decoder, not the forecaster.

## Benchmark

**Data.** PSG4D-HOI (HOI4D videos, 15 fps, 20 s, egocentric, 46 object / 15 predicate
classes), converted with `traser/data/prepare_psg4d_hoi.py`. Split **by object instance**
(`C*/N*`): test objects never appear in training (75 / 10 / 15 by default).

**Protocol.** Evaluation clips are enumerated every second of every test video, for each
(H, F) ∈ {1, 3, 5}²; the headline number is H = 3, F = 3. Every method, including the
training-free baselines, is scored on exactly the same clips (`clip_targets`).

**Metrics** (`forecast/metrics.py`), per future second:

| Metric | Meaning |
|---|---|
| R, mR | recall of (s, pred, o, second) quadruples; mean over predicates |
| P, F1 | precision, so listing every predicate does not pay |
| R@k s | recall at the k-th second ahead (decay over the horizon) |
| mIoU@k s, ADE, FDE | forecast box vs. real box |
| R_loc | SGF-style recall: triplet correct and both boxes IoU ≥ 0.5 |

**Baselines.**

```bash
cd traser && python -m traser_train.forecast.baselines --train data/psg4d_hoi_train.json \
    --test data/psg4d_hoi_test.json --history_sec 3 --horizon_sec 3
```

`copy+copy` and `markov+velocity` receive the ground-truth observed graph (more than TRASER-F
gets), so beating them is a meaningful bar.

**ImparTail / SceneSayer.** Their code is Action-Genome-specific (frame-level boxes, Faster
R-CNN features). Two comparisons:

1. *On PSG4D-HOI*: train SceneSayerSDE (with and without the ImparTail curriculum) on our
   split in the PGAGS/GAGS setting, boxes from our masks, one frame per second; score with
   `recall_at_k` on our quadruples.
2. *On Action Genome*: run TRASER-F with box-shaped masks from AG annotations under the SGA
   protocol (F ∈ {0.3, 0.5, 0.7, 0.9}, R@10/20/50 With / No constraint) and compare to the
   published numbers (SceneSayerSDE, ImparTail, FDSG).

## Running

```bash
# data
python traser/data/prepare_psg4d_hoi.py psg4d --psg4d_json HOI/psg4d_hoi.json \
    --categories HOI/categories.json --hoi4d_root HOI/HOI4D_release --mask_root HOI/HOI4D_mask \
    --out_dir traser/data

# training
STAGE=A bash traser/scripts/train_forecast.sh
STAGE=B bash traser/scripts/train_forecast.sh

# evaluation
cd traser && python -m traser_train.forecast.evaluate --model checkpoints/traser_f_latent_stageB \
    --annotation data/psg4d_hoi_test.json --history_sec 3 --horizon_sec 3 --mode latent \
    --output preds.jsonl

# CPU smoke test (tiny random model, synthetic clip)
python traser/tests/test_forecast_smoke.py
```

## Open points

* **3D.** Boxes are 2D (`forecaster_box_dim=4`). For the 3D variant of TRASER, set the box
  dimension to the 3D box parameterization and compute targets from depth; nothing else in
  the forecaster depends on the box being 2D.
* **Multimodal futures.** The forecaster is deterministic. If H2 works but R@k decays fast
  with k, the next step is a stochastic latent (SDE as in SceneSayer/FDSG) and best-of-N
  scoring.
* **Time resolution.** HOI4D actions are short (grasp ≈ 0.45 s, pickup ≈ 0.4 s), so one
  1-second slot often holds 2–3 consecutive predicates and their order inside the second is
  lost. If the per-second metrics saturate, move to 0.5 s slots (`sample_fps 4`, half-second
  timestamps) — the forecaster is unchanged, only the slot length and the target unit change.
* **Converter assumptions.** The PSG4D-HOI JSON layout is inferred from the authors'
  visualization script; check it on the real files first.
