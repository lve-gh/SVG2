# Inference with TRASER

TRASER is not a video captioner: it takes a video **together with per-object mask
trajectories** and returns a scene graph over exactly those objects. The trajectories are
what ground each predicted object, attribute and relation.

```bash
python traser/inference.py --video clip.mp4 --masks clip_rle.json
```

Weights are pulled from [`UWGZQ/TRASER`](https://huggingface.co/UWGZQ/TRASER) on first use
(~8 GB; set `HF_HOME` to relocate the cache). Pass `--model /path/to/checkpoint` to use a
checkpoint you trained yourself. Progress messages go to stderr; stdout carries only the scene
graph, so `> out.json` captures it directly.

## Worked example

The model repository bundles a video and its trajectory masks. Downloading them and running
the default task takes a couple of minutes on one GPU, most of it loading the weights:

```bash
hf download UWGZQ/TRASER --include "example/*" --local-dir traser/examples
python traser/inference.py \
    --video traser/examples/example/2401075277.mp4 \
    --masks traser/examples/example/2401075277_rle.json \
    --output traser/examples/2401075277.scene_graph.json
```

[`traser/examples/2401075277.reference.json`](../traser/examples/2401075277.reference.json) is
the expected output: 9 objects with attributes and 22 relationships with time spans. It was
produced with the documented environment on an A100; decoding is greedy, so the same
environment reproduces it exactly, while a different GPU or library version may change a few
tokens.

## Inputs

**`--video`** — any format `decord` or `torchcodec` can read. Frames are sampled at ~1 fps
(`BASE_INTERVAL`), at least 4 and at most 128 of them, then resized by the Qwen2.5-VL video
processor. This matches training exactly.

**`--masks`** — per-frame, per-object COCO RLE, one entry per video frame:

```json
[
  [{"size": [H, W], "counts": "..."}, {"size": [H, W], "counts": "..."}],
  [{"size": [H, W], "counts": "..."}, {"size": [H, W], "counts": "..."}]
]
```

Object *k* is column *k* of every frame. This is the format `traser/data/prepare_svg2.py`
writes and the released dataset uses. Objects absent from a frame use an all-zero RLE; the
list may be shorter than the video, in which case the remaining frames count as empty.

**`--masks ... --from_scene_graph`** — read a `stage6_scene_graph.json` produced by
`pipeline/svg2_pipeline.py` instead, and convert its per-object trajectories on the fly. This
is the path from a bare video to a scene graph:

```bash
python pipeline/svg2_pipeline.py --video clip.mp4 --output-dir outputs/
python traser/inference.py --video clip.mp4 \
    --masks outputs/clip/stage6_scene_graph.json --from_scene_graph
```

## What the model is asked

`--task` selects one of the four prompts the model was trained on:

| `--task` | Prompt | Output |
|---|---|---|
| `scene_graph` (default) | "Output the Video Scene Graph …" | objects + attributes + relationships |
| `relationships` | "List all objects and their relationships …" | objects + relationships |
| `attributes` | "List all objects and their attributes …" | objects + attributes |
| `objects` | "List all objects …" | objects only |

## Output

A JSON object printed to stdout (and written to `--output` if given):

```json
{
  "objects": [
    {"object 1": "person", "attributes": ["standing", "wearing a red jacket"]},
    {"object 2": "bicycle", "attributes": ["black", "parked"]}
  ],
  "relationships": [
    [1, "riding", 2, [[0, 2]]],
    [1, "looking at", -1, [[2, 3]]]
  ]
}
```

- `object k` numbers objects **1..K in the order they were passed in**, after objects with no
  mask on any sampled frame have been dropped. `--objects 3 7 9` therefore yields
  `object 1`, `object 2`, `object 3` for mask columns 3, 7 and 9.
- Relationship triplets are `[subject, predicate, object, [[start, end], ...]]`. The interval
  endpoints follow the training annotations: indices into the video sampled at 1 fps, so they
  read as **seconds from the start of the video**, inclusive on both ends. `[[0, 2]]` means
  "during the first three seconds". They are *not* indices into the
  `--temporal_window_length` windows the resamplers use internally.
- A subject or object id of `-1` is the **camera/observer** — it has no object entry.

## Inference options

| Option | Default | Effect |
|---|---|---|
| `--objects` | all | Which mask columns to describe. |
| `--max_objects` | 40 | Cap on the number of objects, matching the released training data. |
| `--coverage_thresh` | 0.5 | Minimum fraction of a token's pixel cell that must be covered by an object's mask for that token to be assigned to it. Lower selects more tokens per object. |
| `--time_reduce` | `max` | How coverage is pooled over the two frames merged into one temporal grid. |
| `--temporal_window_length` | 4 | Seconds per temporal window: how much video one TWR block summarises, and how many `<t - t sec>` markers each object block carries. |
| `--max_new_tokens` | 8192 | Scene graphs for crowded videos are long; truncation shows up as invalid JSON. |

The defaults are the values TRASER was trained with — changing `--coverage_thresh`,
`--time_reduce` or `--temporal_window_length` moves inference away from the training
distribution.

Generation is greedy, so results are reproducible for a given video, mask set and options. The
checkpoint's own `generation_config.json` samples at `temperature=1e-6` — numerically the same
thing — and sets `repetition_penalty=1.05`, which this script inherits.

## Memory and speed

Peak memory is driven by the rearranged sequence length, which grows with the number of
objects and the video duration: each object contributes one OTR block of 32 latents plus, per
non-empty temporal window, a timestamp and 32 TWR latents. A 3B model in bf16 needs roughly 8 GB of weights
plus activations; a 40-object minute-long clip fits comfortably on a 24 GB GPU.

## Using the model directly

`traser/inference.py` deliberately reuses the training modules, so the same three calls give
you the model's inputs:

```python
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_selection import select_tokens      # masks   -> token indices
from traser_train.train.token_arrangement import rearrange_token  # indices -> object blocks

model = TRASER.from_pretrained("UWGZQ/TRASER", torch_dtype=torch.bfloat16).cuda().eval()
```

`select_tokens` returns, per object, the indices of the video tokens its mask covers;
`rearrange_token` replaces the contiguous video-token span with one `<obj_traj_start> …
<obj_traj_end>` block per object and returns `inputs_embeds`, `position_ids`,
`attention_mask` and `rope_deltas` ready for `model.generate`. The model repository on the
Hub also ships a standalone `inference.py` with vendored copies of these modules, for use
without this repository.
