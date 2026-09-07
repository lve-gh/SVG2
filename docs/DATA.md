# SVG2 data

All SVG2 annotations — scene graphs and per-frame segmentation masks — are released in
[**UWGZQ/Synthetic_Visual_Genome2**](https://huggingface.co/datasets/UWGZQ/Synthetic_Visual_Genome2)
as parquet shards. Videos are **not** redistributed: download them from their original
sources (step 2) and convert everything into the training format with
`traser/data/prepare_svg2.py` (step 3).

| Registry name (`--dataset_use`) | `--source` | SVG2 subdirectory | Videos | Mask size |
|---|---|---|---|---|
| `svg2_sav` | `sav` | `data\|masks/cleaned/sav` | SA-V | 44 GB (69 shards) |
| `svg2_pvd` | `pvd` | `data\|masks/cleaned/pvd` | PE-Video | 213 GB (550 shards) |
| `vipseg` | `vipseg` | `data\|masks/academic_datasets/vipseg` | VIPSeg (6 fps) | 0.4 GB |
| `vidor` | `vidor` | `data\|masks/academic_datasets/vidor` | VidOR | 20 GB (23 shards) |
| `vidvrd` | `vidvrd` | `data\|masks/academic_datasets/vidvrd` | ImageNet-VidVRD | 0.1 GB |
| `lvvis` | `lvvis` | `data\|masks/academic_datasets/lvvis` | LV-VIS (5 fps) | 0.1 GB |
| `ovis` | `ovis` | `data\|masks/academic_datasets/ovis` | OVIS (5 fps) | 0.1 GB |



## 1. Download the annotations and masks

All commands run from the repository root; `hf` is installed by `requirements.txt`.

```bash
# Academic splits without VidOR (~0.8 GB) — enough to try the training code
hf download UWGZQ/Synthetic_Visual_Genome2 --repo-type dataset --local-dir SVG2 \
    --include "data/academic_datasets/*" \
              "masks/academic_datasets/vipseg/*" "masks/academic_datasets/vidvrd/*" \
              "masks/academic_datasets/lvvis/*"  "masks/academic_datasets/ovis/*"

# VidOR masks (20 GB)
hf download UWGZQ/Synthetic_Visual_Genome2 --repo-type dataset --local-dir SVG2 \
    --include "masks/academic_datasets/vidor/*"

# SA-V split (44 GB)
hf download UWGZQ/Synthetic_Visual_Genome2 --repo-type dataset --local-dir SVG2 \
    --include "data/cleaned/sav/*" "masks/cleaned/sav/*"

# PE-Video split (213 GB)
hf download UWGZQ/Synthetic_Visual_Genome2 --repo-type dataset --local-dir SVG2 \
    --include "data/cleaned/pvd/*" "masks/cleaned/pvd/*"
```

(With an older CLI, replace `hf download` with `huggingface-cli download`.)

Mask shards are independent: downloading a subset of them and converting works fine —
samples whose masks are missing are reported and skipped. That is the cheapest way to
try the training code before committing to the full download; point the launcher at the
splits you converted, e.g. `DATASETS=vipseg,vidvrd,lvvis,ovis bash traser/scripts/train.sh`.

## 2. Download the videos

Videos are matched to annotations **by file name**: a sample with `video_id = X` must
resolve to a file named `X.mp4` somewhere below the `--video_root` you pass to the
converter (searched recursively, so any layout works).

The masks are aligned to video frames, so each video must have the frame rate below. For
datasets distributed as image frames, re-encode each video's frame folder:

```bash
ffmpeg -framerate <FPS> -pattern_type glob -i '<frames_dir>/*.jpg' \
    -c:v libx264 -pix_fmt yuv420p <video_id>.mp4
```

- **SA-V** (`sav`) — request the SA-V training videos from
  [Meta AI](https://ai.meta.com/datasets/segment-anything-video-downloads/). Already
  named `sav_XXXXXX.mp4` (under `sav_train/sav_XXX/`), native frame rate.
- **PE-Video** (`pvd`) — [facebook/PE-Video](https://huggingface.co/datasets/facebook/PE-Video);
  extract the `.mp4` files (named `<video_id>.mp4`), native frame rate.
- **VIPSeg** (`vipseg`) — frames from the
  [VIPSeg release](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) (720P), re-encoded
  per video folder at **6 fps** to `<video_folder_name>.mp4`.
- **VidOR** (`vidor`) — original training videos from
  [VidOR](https://xdshang.github.io/docs/vidor.html), native frame rate.
- **ImageNet-VidVRD** (`vidvrd`) — videos from
  [ImageNet-VidVRD](https://xdshang.github.io/docs/imagenet-vidvrd.html), native frame rate.
- **LV-VIS** (`lvvis`) — training frames from [LV-VIS](https://github.com/haochenheheda/LVVIS),
  re-encoded per video folder at **5 fps**.
- **OVIS** (`ovis`) — training frames from [OVIS](https://songbai.site/ovis/), re-encoded
  per video folder at **5 fps**.

## 3. Convert to the training format

One command per split, from the repository root. `--out_dir traser/data` writes where the
dataset registry looks by default; any other directory works too, since the annotation JSONs
record absolute paths.

```bash
python traser/data/prepare_svg2.py --source vipseg --svg2_root SVG2 --video_root /path/to/VIPSeg_videos_6fps --out_dir traser/data
python traser/data/prepare_svg2.py --source vidor  --svg2_root SVG2 --video_root /path/to/VidOR/video       --out_dir traser/data
python traser/data/prepare_svg2.py --source vidvrd --svg2_root SVG2 --video_root /path/to/vidvrd-videos     --out_dir traser/data
python traser/data/prepare_svg2.py --source lvvis  --svg2_root SVG2 --video_root /path/to/LVVIS_videos_5fps --out_dir traser/data
python traser/data/prepare_svg2.py --source ovis   --svg2_root SVG2 --video_root /path/to/OVIS_videos_5fps  --out_dir traser/data
python traser/data/prepare_svg2.py --source sav    --svg2_root SVG2 --video_root /path/to/sav_train         --out_dir traser/data
python traser/data/prepare_svg2.py --source pvd    --svg2_root SVG2 --video_root /path/to/PE-Video          --out_dir traser/data --max_object 40
```

This gives the layout the training code expects:

```
traser/data/
├── svg2_sav.json                       # annotation list, one entry per sample
├── svg2_pvd.json
├── vipseg.json  vidor.json  vidvrd.json  lvvis.json  ovis.json
└── masks/
    ├── sav/<video_id>_rle.json         # per-frame, per-object COCO RLE
    ├── pvd/<video_id>_rle.json
    └── vipseg/ vidor/ vidvrd/ lvvis/ ovis/
```

Notes:

- `--max_object 40` on `pvd` reproduces the released TRASER training data (oversized
  samples keep their first 40 objects; relationships touching dropped objects are removed).
- Camera relationships — edges whose subject or object is the camera/observer id `-1` — are
  kept whenever their other endpoint is a kept object.
- The converter unpacks the mask parquets into one RLE JSON per video and is resumable, so
  interrupted runs can simply be restarted. The unpacked masks take roughly 2–4× the
  parquet size on disk.

## Annotation format

Each annotation JSON is a list of samples:

```json
{
  "video": "/path/to/LV-VIS/videos_fps5/00040.mp4",
  "mask_json": "/path/to/masks/lvvis/00040_rle.json",
  "obj_list": [0],
  "has_relationships": false,
  "conversations": [
    {"from": "human", "value": "List all objects from the video and object trajectories:\n<video>\n"},
    {"from": "gpt", "value": "{\"objects\":[{\"object_0\":\"handsaw\"}]}"}
  ]
}
```

- `mask_json` holds per-frame, per-object RLE masks in COCO `pycocotools` format:
  `list(frames)` of `list(objects)` of `{"size": [H, W], "counts": ...}`, one entry per
  **video frame**. Objects absent from a frame use an all-zero RLE; masks may cover fewer
  frames than the video, in which case the remaining frames are treated as empty.
- `obj_list` are the object ids used for training; each id indexes into the per-frame mask
  list. Ids may be non-contiguous — they always match the `object_N` keys of the assistant
  target.
- The assistant target is a JSON scene graph. Relationship triplets are
  `[subject_id, predicate, object_id, [[start, end], ...]]`. The interval endpoints are indices
  into the video sampled at 1 fps, so they read as **seconds from the start of the video**,
  inclusive on both ends. A subject/object id of `-1` denotes the camera/observer; it has no
  object entry or mask, and the training pipeline keeps it unchanged when the other ids are
  renumbered.
- The prompt depends on what the sample annotates: relationships + attributes →
  `"Output the Video Scene Graph ..."`; relationships only → `"List all objects and their
  relationships ..."`; attributes only → `"List all objects and their attributes ..."`;
  objects only → `"List all objects ..."`. Samples without relationships omit the
  `relationships` key and set `has_relationships: false`.

## Reading the masks

An object absent from a frame is stored as an all-zero RLE. Skip those by area instead of
decoding them.

```python
if maskUtils.area(rle) == 0:
    continue          # object absent on this frame
mask = maskUtils.decode(rle)
```

`traser_train/data/data_qwen.py` and `traser/inference.py` already do this. `prepare_svg2.py`
copies the released RLEs verbatim, so `traser/data/masks/` stays a faithful copy of the
dataset.
