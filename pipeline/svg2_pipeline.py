#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG2 — Video Scene Graph generation pipeline (single-file, local, open-source release).

Given a single input video, the pipeline produces, for every salient object, a mask
**trajectory** across the video together with a structured scene-graph record
(``name``, ``attributes``, ``relationships``, ``actions``) plus the free-form description
it was derived from, and the temporal/spatial relationships between the named objects.

It runs as six sequential stages, each of which reads a well-defined artifact from disk
(or the raw video, for stage 1) and writes a well-defined artifact, so a run can start and
stop at any stage and resume later:

    1. maskgen       - automatic "segment everything" mask generation on sampled frames
    2. track         - propagate the seed masks through the whole video (two-pass, with
                       mid-video re-discovery of newly appearing objects)
    3. cleanup       - de-duplicate near-identical/contained tracks and morphologically
                       clean masks
    4. caption       - describe each tracked object with a region-captioning VLM (NVIDIA DAM)
    5. structure     - turn each description into a strict ``{Object, Attributes,
                       Relationships, Actions}`` record with the OpenAI API
    6. relationships - infer temporal + spatial relationships between the named objects
                       with two OpenAI vision calls

Mask generation and tracking both use the **official ``sam2`` package**, which has no
``transformers`` dependency; that is what lets the whole pipeline share one environment with
the transformers-4.x-only DAM captioner. SAM 3 is transformers-5.x-only and is therefore not
available here. All model weights are downloaded from the Hugging Face Hub on first use;
nothing is read from local checkpoints.

Example
-------
Run the whole pipeline with SAM 2::

    export OPENAI_API_KEY=sk-...
    python svg2_pipeline.py --video path/to/clip.mp4 --output-dir outputs/

Run only stages 1-3 (no captioner / API key needed) with the small SAM 2 model::

    python svg2_pipeline.py --video clip.mp4 --output-dir outputs/ \
        --maskgen-model-id facebook/sam2.1-hiera-tiny \
        --tracking-model-id facebook/sam2.1-hiera-tiny \
        --start-stage 1 --end-stage 3

Resume from a previous run (reads the stage-3 artifact, runs captioning + structuring)::

    python svg2_pipeline.py --video clip.mp4 --output-dir outputs/ --start-stage 4

See ``python svg2_pipeline.py --help`` for the full CLI, or ``--dump-config`` to write the
effective configuration to a YAML file you can edit and pass back with ``--config``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("svg2")

# --------------------------------------------------------------------------------------
# Stage registry
# --------------------------------------------------------------------------------------

#: Canonical stage order. The CLI / config accept either a stage name or its 1-based number.
STAGE_NAMES: list[str] = ["maskgen", "track", "cleanup", "caption", "structure", "relationships"]
STAGE_TO_INDEX: dict[str, int] = {name: i + 1 for i, name in enumerate(STAGE_NAMES)}
#: Artifact file written by each stage, relative to ``<output_dir>/<video_stem>/``.
STAGE_ARTIFACT: dict[int, str] = {
    1: "stage1_masks.json",
    2: "stage2_tracks.json",
    3: "stage3_tracks_clean.json",
    4: "stage4_descriptions.json",
    5: "stage5_scene_graph.json",
    6: "stage6_scene_graph.json",
}

#: Default Hugging Face repo id for the SAM 2 weights (used by both mask generation and tracking).
DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"


def resolve_stage(value: "int | str") -> int:
    """Map a stage name (``"track"``) or 1-based number (``2``/``"2"``) to its index."""
    if isinstance(value, int):
        idx = value
    elif isinstance(value, str) and value.isdigit():
        idx = int(value)
    elif isinstance(value, str) and value in STAGE_TO_INDEX:
        idx = STAGE_TO_INDEX[value]
    else:
        raise ValueError(f"Unknown stage {value!r}; use one of {STAGE_NAMES} or 1..5")
    if not 1 <= idx <= len(STAGE_NAMES):
        raise ValueError(f"Stage index {idx} out of range (1..{len(STAGE_NAMES)})")
    return idx


# ======================================================================================
# Section 1 — Configuration
# ======================================================================================


@dataclass
class IOConfig:
    """Input/output locations."""

    video_path: str = ""
    """Path to the input video (required when ``start_stage == 1``)."""
    output_dir: str = "outputs"
    """Root directory for artifacts. Each video gets its own ``<output_dir>/<video_stem>/``."""
    cache_dir: str = str(Path(__file__).resolve().parent / ".cache")
    """Directory for downloaded models (used as ``HF_HOME``) and other cached files."""
    overwrite: bool = False
    """If ``True``, recompute and overwrite stage artifacts even when they already exist."""


@dataclass
class RuntimeConfig:
    """Hardware / execution settings shared by every stage."""

    device: str = "cuda"
    """Torch device for all GPU models (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``)."""
    log_level: str = "INFO"


@dataclass
class MaskGenConfig:
    """Stage 1 — automatic ("segment everything") mask generation.

    This stage always uses **SAM 2** (SAM 3 has no automatic mask generator), via the
    official ``SAM2AutomaticMaskGenerator``. Hyperparameter defaults mirror the original
    configuration: a *multi-scale point grid* (one grid per crop layer, listed in
    ``grid_points_per_side``) combined with cropped layers.
    """

    model_id: str = "facebook/sam2.1-hiera-large"
    """Hugging Face repo id for the SAM 2 weights used by the automatic mask generator."""
    frame_sample_rate: int = 20
    """Run mask generation on every Nth frame."""
    grid_points_per_side: list[int] = field(default_factory=lambda: [32, 16, 4])
    """Points-per-side for each crop layer's grid. ``[32, 16, 4]`` reproduces the original
    multi-scale grids; the number of crop layers is ``len(grid_points_per_side) - 1``."""
    points_per_batch: int = 64
    """Number of grid points forwarded through the decoder at once."""
    pred_iou_thresh: float = 0.75
    """Discard masks whose predicted IoU (quality) is below this."""
    stability_score_thresh: float = 0.85
    """Discard masks whose stability score is below this."""
    stability_score_offset: float = 1.0
    """Cutoff offset used when computing the stability score."""
    box_nms_thresh: float = 0.7
    """IoU threshold for non-max suppression of mask bounding boxes."""
    crop_nms_thresh: float = 0.7
    """IoU threshold for non-max suppression across crops."""
    crop_overlap_ratio: float = 0.5
    """Fractional overlap between neighbouring crops."""
    min_mask_region_area: int = 200
    """Drop mask regions smaller than this many pixels (post-processing)."""
    use_m2m: bool = True
    """Use one step of mask-to-mask refinement (matches the original configuration)."""
    max_overlap_ratio: float = 0.9
    """In the max-non-overlapping filter, keep a mask only if the fraction of its area
    already covered by larger kept masks is below this."""


@dataclass
class TrackingConfig:
    """Stage 2 — video propagation and mid-video object re-discovery.

    Tracking uses the official SAM 2 video predictor (``sam2`` package), which has no
    transformers dependency and therefore coexists with the transformers 4.x that the DAM
    captioner requires. It always runs two passes: a discovery pass that finds objects (and
    when they first appear), then a clean pass that re-seeds every object at its
    first-appearance frame for consistent trajectories.
    """

    model_id: str = DEFAULT_SAM2_MODEL_ID
    """Hugging Face repo id for the SAM 2 video predictor weights."""
    offload_to_cpu: bool = True
    """Offload video frames and inference state to CPU to reduce GPU memory use."""
    max_objects: int = 65
    """Stop adding newly discovered objects once this many are being tracked."""
    rediscovery_enabled: bool = True
    """Detect and add objects that appear after the first frame."""
    rediscovery_overlap_thresh: float = 0.1
    """Add new objects only if newly detected masks cover at least this fraction of the
    currently untracked region."""
    rediscovery_min_untracked_frac: float = 0.05
    """Only attempt re-discovery when the untracked region is at least this fraction of the frame."""
    match_iou_thresh: float = 0.9
    """A stage-1 detection counts as already-tracked if it overlaps a tracked mask by this much."""
    adaptive_sample_rate: bool = True
    """Scale the re-discovery check stride by ``total_frames // 100 + 1`` for long videos."""


@dataclass
class CleanupConfig:
    """Stage 3 — track de-duplication and morphological cleanup."""

    enabled: bool = True
    contain_ratio_thresh: float = 0.9
    """Per-frame containment ratio above which one track is considered inside another."""
    coverage_share_thresh: float = 0.7
    """Fraction of overlapping frames that must be 'contained' to trigger removal."""
    min_overlap_frames: int = 5
    """Minimum number of co-occurring frames before two tracks are compared."""
    morph_kernel: int = 3
    """Kernel side length for the morphological open (erode then dilate)."""
    morph_iterations: int = 1
    """Number of erode/dilate iterations."""
    morph_min_area: int = 32
    """Drop a per-frame mask if it falls below this area after the open."""


@dataclass
class CaptionConfig:
    """Stage 4 — per-object region captioning with NVIDIA Describe-Anything.

    Defaults mirror the original DAM-3B-Video invocation.
    """

    model_id: str = "nvidia/DAM-3B-Video"
    """Hugging Face repo id for the Describe-Anything model."""
    prompt_mode: str = "focal_prompt"
    """DAM prompt mode (mapped internally to ``full+focal_crop``)."""
    conv_mode: str = "v1"
    """DAM conversation template."""
    max_objects: int = 40
    """Describe at most this many objects, chosen by largest total mask area."""
    frames_per_object: int = 8
    """Number of frames sampled per object and fed to the captioner."""
    frame_sampling: str = "uniform"
    """How to sample an object's frames: ``"uniform"`` or ``"max_area"``."""
    temperature: float = 0.2
    top_p: float = 0.5
    num_beams: int = 1
    max_new_tokens: int = 512
    prompt: str = (
        "Video: {image_tokens}\nGiven the video in the form of a sequence of frames above, "
        "describe the object in the masked region in the video in detail."
    )
    """Captioning prompt. ``{image_tokens}`` is replaced by one ``<image>`` token per frame."""


@dataclass
class StructureConfig:
    """Stage 5 — structured scene-graph extraction via the OpenAI API.

    The model returns a strict ``{Object, Attributes, Relationships, Actions}`` JSON object.
    Requires the API key to be present in the environment variable named by ``api_key_env``.
    """

    enabled: bool = True
    model_id: str = "gpt-5-mini"
    """OpenAI model id (e.g. ``gpt-5-mini``, ``gpt-5-nano``, ``gpt-4o-mini``)."""
    api_key_env: str = "OPENAI_API_KEY"
    """Name of the environment variable holding the API key."""
    base_url: Optional[str] = None
    """Optional base URL for OpenAI-compatible endpoints; ``None`` uses the default OpenAI API."""
    temperature: Optional[float] = None
    """Sampling temperature. ``None`` omits the parameter (some reasoning models reject it)."""
    max_retries: int = 3
    """Number of attempts before giving up on a single description."""


@dataclass
class RelationshipConfig:
    """Stage 6 — temporal + spatial relationship extraction via the OpenAI API (vision).

    Two direct API calls per video: one for temporal (non-spatial) relationships and one for
    spatial relationships. Each call sends the frames (sampled at ~1 fps) plus, per frame, the
    list of named objects and their bounding boxes, and returns relationship tuples that
    reference object ids. Requires the API key in the environment variable ``api_key_env``.
    """

    enabled: bool = True
    model_id: str = "gpt-5"
    """OpenAI vision model id. The reference pipeline used full GPT-5 (``gpt-5-2025-08-07``) for
    relationship extraction, not a mini/nano model."""
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    extract_temporal: bool = True
    """Run the temporal (non-spatial) relationship call."""
    extract_spatial: bool = True
    """Run the spatial relationship call."""
    frame_sample_fps: float = 1.0
    """Sample the video at roughly this many frames per second for relationship reasoning."""
    max_frames: int = 24
    """Upper bound on frames sent per call; uniformly sub-sample if the 1-fps set exceeds it."""
    image_detail: str = "high"
    """OpenAI vision image detail: ``"high"``, ``"low"`` or ``"auto"``."""
    reasoning_effort: Optional[str] = "medium"
    """Reasoning effort sent to the model; models that reject it fall back automatically."""
    max_completion_tokens: int = 8192
    keep_uncertain: bool = False
    """If ``False``, objects whose label contains "uncertain" are omitted from the prompt."""
    max_retries: int = 3


@dataclass
class PipelineConfig:
    """Top-level configuration composed of the per-stage sub-configs."""

    io: IOConfig = field(default_factory=IOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mask_gen: MaskGenConfig = field(default_factory=MaskGenConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)

    start_stage: int = 1
    end_stage: int = 6
    save_stages: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    """Stages whose intermediate artifact is written to disk. The ``end_stage`` artifact is
    always written regardless of this list."""

    # ---- derived / helpers ---------------------------------------------------------

    def should_save(self, stage_idx: int) -> bool:
        """Whether the artifact for ``stage_idx`` should be persisted."""
        return stage_idx in self.save_stages or stage_idx == self.end_stage

    def normalize(self) -> "PipelineConfig":
        """Validate and canonicalize stage selectors. Returns ``self`` for chaining."""
        self.start_stage = resolve_stage(self.start_stage)
        self.end_stage = resolve_stage(self.end_stage)
        if self.start_stage > self.end_stage:
            raise ValueError(f"start_stage ({self.start_stage}) > end_stage ({self.end_stage})")
        self.save_stages = sorted({resolve_stage(s) for s in self.save_stages})
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_file(self, path: "str | Path") -> None:
        """Write the effective config to a ``.yaml`` or ``.json`` file."""
        path = Path(path)
        data = self.to_dict()
        text = _dump_yaml(data) if path.suffix in {".yaml", ".yml"} else json.dumps(data, indent=2)
        path.write_text(text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Build a config from a nested dict, filling unspecified fields with defaults."""
        sub = {
            "io": IOConfig,
            "runtime": RuntimeConfig,
            "mask_gen": MaskGenConfig,
            "tracking": TrackingConfig,
            "cleanup": CleanupConfig,
            "caption": CaptionConfig,
            "structure": StructureConfig,
            "relationship": RelationshipConfig,
        }
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                logger.warning("Ignoring unknown config key %r", key)
                continue
            if key in sub and isinstance(value, dict):
                kwargs[key] = sub[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: "str | Path") -> "PipelineConfig":
        path = Path(path)
        text = path.read_text()
        data = _load_yaml(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        return cls.from_dict(data)


def _load_yaml(text: str) -> dict[str, Any]:
    import yaml  # local import: only needed when a YAML config is used

    return yaml.safe_load(text) or {}


def _dump_yaml(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


# ======================================================================================
# Section 2 — RLE / IO / video helpers
# ======================================================================================


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a 2-D binary mask as a COCO RLE dict with a JSON-serialisable ``counts`` string."""
    import pycocotools.mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a 2-D boolean mask."""
    import pycocotools.mask as mask_utils

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def rle_area(rle: Optional[dict]) -> int:
    """Pixel area of an RLE mask (0 for ``None`` / empty placeholders)."""
    import pycocotools.mask as mask_utils

    if not rle or not rle.get("counts"):
        return 0
    return int(mask_utils.area(rle))


def empty_rle(height: int, width: int) -> dict:
    """An all-zero RLE mask of the given size."""
    return encode_rle(np.zeros((height, width), dtype=np.uint8))


def read_json(path: "str | Path") -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: "str | Path", data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def video_stem(video_path: str) -> str:
    return Path(video_path).stem


def artifact_path(cfg: PipelineConfig, stage_idx: int) -> Path:
    """Absolute path of the artifact written by ``stage_idx`` for the current video."""
    stem = video_stem(cfg.io.video_path)
    return Path(cfg.io.output_dir) / stem / STAGE_ARTIFACT[stage_idx]


def read_video_frames(video_path: str) -> tuple[list[np.ndarray], float, int, int]:
    """Decode an entire video to a list of RGB ``uint8`` frames.

    Returns ``(frames, fps, height, width)``.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    height, width = frames[0].shape[:2]
    logger.info("Decoded %d frames (%dx%d, %.1f fps) from %s", len(frames), width, height, fps, video_path)
    return frames, fps, height, width


def _add_local_packages(cache_dir: str) -> None:
    """Make a ``<cache_dir>/pylibs`` directory importable, if present.

    This is a convenience for environments where optional dependencies (e.g. ``sam2``) were
    installed into the cache via ``pip install --target`` instead of the active environment.
    For a normal ``pip install -r requirements.txt`` the directory does not exist and this is
    a no-op.
    """
    import sys

    pylibs = Path(cache_dir) / "pylibs"
    if pylibs.is_dir() and str(pylibs) not in sys.path:
        sys.path.insert(0, str(pylibs))


# ======================================================================================
# Section 3 — Mask & track geometry utilities (model-free)
# ======================================================================================


def select_max_non_overlapping(masks: list[np.ndarray], max_overlap_ratio: float = 0.9) -> list[np.ndarray]:
    """Greedily keep the largest masks that add new coverage.

    Masks are sorted by area (descending); a mask is kept only if the fraction of its own
    area already covered by previously kept masks is below ``max_overlap_ratio``. This is a
    faithful port of the original ``MaskFilter.max_non_overlapping_masks``.
    """
    if not masks:
        return []
    areas = np.array([int(m.sum()) for m in masks])
    order = np.argsort(-areas)
    union = np.zeros_like(masks[0], dtype=bool)
    kept: list[np.ndarray] = []
    for idx in order:
        if areas[idx] == 0:
            continue
        overlap = int(np.logical_and(union, masks[idx]).sum())
        if overlap / float(areas[idx]) < max_overlap_ratio:
            kept.append(masks[idx])
            union |= masks[idx]
    return kept


def match_detections_to_tracks(
    tracked_ids: list[int],
    tracked_masks: list[np.ndarray],
    detection_masks: list[np.ndarray],
    iou_thresh: float = 0.9,
) -> list[int]:
    """Assign each detection either an existing track id (if it overlaps one strongly) or a
    fresh id. Overlap is detection-area-normalised, matching the original behaviour.
    """
    existing = set(tracked_ids)
    next_id = (max(existing) + 1) if existing else 0
    assigned: list[int] = []
    for det in detection_masks:
        det_area = int(det.sum())
        best_overlap, best_id = 0.0, None
        for tid, tmask in zip(tracked_ids, tracked_masks):
            inter = int(np.logical_and(tmask, det).sum())
            ratio = inter / det_area if det_area > 0 else 0.0
            if ratio > best_overlap:
                best_overlap, best_id = ratio, tid
        if best_overlap >= iou_thresh and best_id is not None:
            assigned.append(best_id)
        else:
            assigned.append(next_id)
            next_id += 1
    return assigned


def _track_length(track: list[Optional[dict]]) -> int:
    return sum(1 for r in track if rle_area(r) > 0)


def _track_median_area(track: list[Optional[dict]]) -> float:
    areas = [rle_area(r) for r in track if rle_area(r) > 0]
    return float(np.median(areas)) if areas else 0.0


def _rle_iou(a: dict, b: dict) -> float:
    import pycocotools.mask as mask_utils

    # iou() returns a (len(dt), len(gt)) matrix; index the single pair explicitly.
    return float(mask_utils.iou([a], [b], [0])[0, 0])


def _track_recent_iou(track: list[Optional[dict]], lookback: int = 5) -> float:
    idxs = [i for i, r in enumerate(track) if rle_area(r) > 0]
    if len(idxs) < 2:
        return 0.0
    idxs = idxs[-(lookback + 1):]
    ious = [_rle_iou(track[a], track[b]) for a, b in zip(idxs[:-1], idxs[1:])]
    return float(np.mean(ious)) if ious else 0.0


def track_stability_score(track: list[Optional[dict]]) -> float:
    """Stability heuristic used to choose a survivor among duplicate tracks (ported verbatim)."""
    return 0.5 * _track_length(track) + 0.05 * _track_median_area(track) + 200.0 * _track_recent_iou(track, 5)


def _containment_per_frame(track_a: list[Optional[dict]], track_b: list[Optional[dict]]) -> list[tuple[float, float]]:
    import pycocotools.mask as mask_utils

    out: list[tuple[float, float]] = []
    for ra, rb in zip(track_a, track_b):
        a, b = rle_area(ra), rle_area(rb)
        if a == 0 or b == 0:
            continue
        inter = int(mask_utils.area(mask_utils.merge([ra, rb], intersect=True)))
        out.append((inter / max(1, a), inter / max(1, b)))
    return out


def dedupe_tracks(
    tracks: list[list[Optional[dict]]],
    contain_ratio_thresh: float = 0.9,
    coverage_share_thresh: float = 0.7,
    min_overlap_frames: int = 5,
) -> list[bool]:
    """Decide which tracks survive de-duplication.

    Returns a boolean ``alive`` flag per input track. A track is removed when it is mostly
    contained in another across enough co-occurring frames; ties are broken by
    :func:`track_stability_score`. Faithful port of ``dedupe_tracks_preserve_K`` (here we
    return the alive mask instead of padding removed tracks with placeholders).
    """
    n = len(tracks)
    alive = [True] * n
    scores = [track_stability_score(t) for t in tracks]
    order = sorted(range(n), key=lambda k: -_track_length(tracks[k]))
    for i, a in enumerate(order):
        if not alive[a]:
            continue
        for b in order[i + 1:]:
            if not alive[b]:
                continue
            per = _containment_per_frame(tracks[a], tracks[b])
            if len(per) < min_overlap_frames:
                continue
            strong_a = sum(1 for ca, _ in per if ca >= contain_ratio_thresh) / len(per)
            strong_b = sum(1 for _, cb in per if cb >= contain_ratio_thresh) / len(per)
            kill_a = kill_b = False
            if strong_a >= coverage_share_thresh and strong_b < coverage_share_thresh:
                kill_a = True
            elif strong_b >= coverage_share_thresh and strong_a < coverage_share_thresh:
                kill_b = True
            elif strong_a >= coverage_share_thresh and strong_b >= coverage_share_thresh:
                kill_b = scores[a] >= scores[b]
                kill_a = not kill_b
            if kill_a:
                alive[a] = False
                break
            if kill_b:
                alive[b] = False
    return alive


def morphological_open(mask: np.ndarray, kernel: int, iterations: int) -> np.ndarray:
    """Erode then dilate a binary mask to remove thin spurs / speckle."""
    import cv2

    k = np.ones((kernel, kernel), np.uint8)
    out = mask.astype(np.uint8)
    for _ in range(max(1, iterations)):
        out = cv2.morphologyEx(out, cv2.MORPH_ERODE, k, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_DILATE, k, iterations=1)
    return out.astype(bool)


def mask_bbox(mask: np.ndarray) -> Optional[list[int]]:
    """Tight ``[x1, y1, x2, y2]`` bounding box of a binary mask, or ``None`` if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ======================================================================================
# Section 4 — Segmentation backbones (SAM 2 / SAM 3)
# ======================================================================================


def make_point_grid(points_per_side: int) -> np.ndarray:
    """A uniform grid of ``points_per_side**2`` normalised ``(x, y)`` points in ``[0, 1]``."""
    coords = np.linspace(0, 1, points_per_side)
    xs = np.tile(coords[None, :], (points_per_side, 1))
    ys = np.tile(coords[:, None], (1, points_per_side))
    return np.stack([xs.flatten(), ys.flatten()], axis=1)


class AutomaticMaskGenerator:
    """"Segment everything" mask generation using the official ``SAM2AutomaticMaskGenerator``.

    Stage 1 always uses SAM 2: SAM 3 has no automatic mask generator, and the multi-scale
    crop+grid configuration this pipeline relies on is not reproducible through the
    transformers mask-generation pipeline (which cannot stack differently-sized crops). The
    SAM 2 weights are pulled from the Hugging Face Hub by ``model_id``.
    """

    def __init__(self, cfg: MaskGenConfig, device: str) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf

        self.cfg = cfg
        self.device = device
        model = build_sam2_hf(cfg.model_id, device=device)
        point_grids = [make_point_grid(n) for n in cfg.grid_points_per_side]
        self._generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=None,  # use the explicit multi-scale point_grids instead
            point_grids=point_grids,
            points_per_batch=cfg.points_per_batch,
            pred_iou_thresh=cfg.pred_iou_thresh,
            stability_score_thresh=cfg.stability_score_thresh,
            stability_score_offset=cfg.stability_score_offset,
            box_nms_thresh=cfg.box_nms_thresh,
            crop_n_layers=len(cfg.grid_points_per_side) - 1,
            crop_nms_thresh=cfg.crop_nms_thresh,
            crop_overlap_ratio=cfg.crop_overlap_ratio,
            min_mask_region_area=cfg.min_mask_region_area,
            use_m2m=cfg.use_m2m,
            multimask_output=False,
        )

    def generate(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Return a list of boolean HxW masks for one RGB frame."""
        import torch
        from contextlib import nullcontext

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            records = self._generator.generate(frame_rgb)
        return [r["segmentation"].astype(bool) for r in records if int(r["segmentation"].sum()) > 0]


class VideoTracker:
    """Mask-prompted video object tracker using the official SAM 2 video predictor.

    Uses the official ``sam2`` package (``build_sam2_video_predictor_hf``), which has no
    transformers dependency and therefore coexists with the transformers 4.x required by the
    DAM captioner. Weights are pulled from the Hugging Face Hub by ``model_id``.

    API: ``init_state(video_path)`` -> ``add_mask(state, frame_idx, obj_id, mask)`` ->
    iterate ``propagate(state, start_frame, max_frames)`` -> ``reset(state)``.
    """

    def __init__(self, model_id: str, device: str, offload_to_cpu: bool = True) -> None:
        from sam2.build_sam import build_sam2_video_predictor_hf

        self.device = device
        self.offload = offload_to_cpu
        self.predictor = build_sam2_video_predictor_hf(model_id, device=device)

    def init_state(self, video_path: str):
        """Initialise tracking state by encoding the video at ``video_path``."""
        return self.predictor.init_state(
            video_path,
            offload_video_to_cpu=self.offload,
            offload_state_to_cpu=self.offload,
        )

    def reset(self, state) -> None:
        self.predictor.reset_state(state)

    def add_mask(self, state, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        """Seed/condition object ``obj_id`` with a binary ``mask`` at ``frame_idx``."""
        self.predictor.add_new_mask(state, int(frame_idx), int(obj_id), mask.astype(np.uint8))

    def propagate(self, state, start_frame: int, max_frames: Optional[int]):
        """Yield ``(frame_idx, {obj_id: boolean mask})`` forward from ``start_frame``."""
        import torch
        from contextlib import nullcontext

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                state, start_frame_idx=start_frame, max_frame_num_to_track=max_frames
            ):
                binary = (mask_logits > 0.0).cpu().numpy()
                if binary.ndim == 4:  # (num_objects, 1, H, W) -> (num_objects, H, W)
                    binary = binary[:, 0]
                yield int(frame_idx), {int(o): binary[i].astype(bool) for i, o in enumerate(obj_ids)}


# ======================================================================================
# Section 5 — Region captioner (NVIDIA Describe-Anything)
# ======================================================================================


class DamCaptioner:
    """Per-object region captioner wrapping NVIDIA's Describe-Anything (DAM) model.

    DAM is a custom package (not pure transformers); weights download from the HF Hub.
    """

    _DAM_PROMPT_MODES = {"focal_prompt": "full+focal_crop"}

    def __init__(self, cfg: CaptionConfig, device: str) -> None:
        try:
            from dam import DescribeAnythingModel, disable_torch_init
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "The Describe-Anything package is required for stage 4. Install with:\n"
                "    pip install git+https://github.com/NVlabs/describe-anything.git"
            ) from exc

        disable_torch_init()
        self.cfg = cfg
        self.model = DescribeAnythingModel(
            model_path=cfg.model_id,
            conv_mode=cfg.conv_mode,
            prompt_mode=self._DAM_PROMPT_MODES.get(cfg.prompt_mode, cfg.prompt_mode),
        ).to(device)

    def describe(self, frames_rgb: list[np.ndarray], masks: list[np.ndarray]) -> str:
        """Describe the masked object given a list of RGB frames and aligned binary masks."""
        from PIL import Image

        images = [Image.fromarray(f) for f in frames_rgb]
        mask_images = [Image.fromarray((m.astype(np.uint8) * 255)) for m in masks]
        image_tokens = " ".join(["<image>"] * len(images))
        query = self.cfg.prompt.format(image_tokens=image_tokens)
        return self.model.get_description(
            images,
            mask_images,
            query,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            num_beams=self.cfg.num_beams,
            max_new_tokens=self.cfg.max_new_tokens,
        )


# ======================================================================================
# Section 6 — Structured scene-graph extraction (OpenAI API)
# ======================================================================================

#: System prompt for the structured-extraction model.
STRUCTURE_SYSTEM_PROMPT = "You are an information-extraction engine for visual scene understanding."

#: Strict JSON schema for an object's scene-graph record (OpenAI structured outputs). Field
#: names are lower-case to match the user prompt below.
SCENE_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "relationships": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["object", "attributes", "relationships", "actions"],
    "additionalProperties": False,
}


def build_structure_prompt(description: str) -> str:
    """User prompt asking the LLM to extract a structured record from a free-form description."""
    return f"""You are an expert in scene understanding. I will give you a short paragraph that describes a video clip.Your task is to extract structured information about a single object described in the paragraph.Please return a JSON with the following fields:

"object": The main object being described (e.g., "person", "dog", "car"). If the inference about the object is uncertain based on the description, add "(uncertain)" after the object name.

"attributes": A list of ONLY the visual/physical attributes that can be directly observed about the object itself. Include only:

Visual appearance: color, shape, size, texture, pattern, material appearance, style

Physical properties: state, transparency, reflectiveness, orientation, material

Design elements: stripes, dots, logos, decorative featuresDO NOT include: implied states, inferred conditions, functional descriptions, or anything that describes the object's interaction with its environment.

"relationships": A list of relationships between this object and other entities or the environment (e.g., "on top of table", "next to person", "inside container", "facing camera", "part of group").

"actions": A list of actions that the object is performing or movements it is making (e.g., "rotating", "moving", "falling", "bouncing", "sliding").

Important distinctions:

Attributes = What the object looks like (visual only). Please use ADJECTIVE form

Relationships = How the object relates to other things spatially, functionally, or contextually

Actions = What the object is doing or how it's moving

Now process the following description: \"\"\"{description}\"\"\"."""


class SceneGraphStructurer:
    """Convert a free-form object description into a strict scene-graph record."""

    def __init__(self, cfg: StructureConfig) -> None:
        from openai import OpenAI

        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {cfg.api_key_env} is not set; it is required for stage 5."
            )
        self.cfg = cfg
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def structure(self, description: str) -> dict[str, Any]:
        """Return the canonical ``{Object, Attributes, Relationships, Actions}`` record."""
        messages = [
            {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
            {"role": "user", "content": build_structure_prompt(description)},
        ]
        extra: dict[str, Any] = {}
        if self.cfg.temperature is not None:
            extra["temperature"] = self.cfg.temperature

        last_error: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model_id,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "scene_graph", "schema": SCENE_GRAPH_SCHEMA, "strict": True},
                    },
                    **extra,
                )
                return _validate_scene_graph(json.loads(response.choices[0].message.content))
            except Exception as exc:  # noqa: BLE001 - retry on any API/parse error
                last_error = exc
                logger.warning("Structuring attempt %d/%d failed: %s", attempt + 1, self.cfg.max_retries, exc)
                # Fall back to plain JSON mode if structured outputs are unsupported.
                if attempt == 0:
                    extra["response_format"] = {"type": "json_object"}  # type: ignore[assignment]
        raise RuntimeError(f"Failed to structure description after {self.cfg.max_retries} attempts") from last_error


def _validate_scene_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce the model's (lower-case-keyed) JSON into the canonical capitalised record shape.

    Accepts either lower-case keys (as the prompt requests) or capitalised keys, for robustness.
    """
    def pick(*keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    return {
        "Object": str(pick("object", "Object") or ""),
        "Attributes": [str(x) for x in (pick("attributes", "Attributes") or [])],
        "Relationships": [str(x) for x in (pick("relationships", "Relationships") or [])],
        "Actions": [str(x) for x in (pick("actions", "Actions") or [])],
    }


# ======================================================================================
# Section 6b — Relationship extraction (OpenAI vision API)
# ======================================================================================
#
# Two direct API calls per video reproduce the original batch pipeline's temporal and spatial
# relationship extraction. The prompts and JSON schemas below are copied verbatim from that
# codebase so the model behaviour is unchanged; only the delivery mechanism (batch -> two
# regular calls) and the input assembly (local frames -> base64 data URLs) differ.

#: Spatial-relationship JSON schema (verbatim): tuples [subject_id, predicate, object_id, [[start,end],...]].
SPATIAL_RELATION_SCHEMA: dict[str, Any] = {
    "name": "video_scene_graph_spatial",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "array",
                    "description": "[subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...]]",
                    "minItems": 4,
                    "maxItems": 4,
                    "prefixItems": [
                        {"type": "integer", "description": "subject_id"},
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                            "description": "concise spatial relationship string, no underscores"
                        },
                        {"type": "integer", "description": "object_id"},
                        {
                            "type": "array",
                            "minItems": 1,
                            "description": "list of [start, end] inclusive intervals (frames 0..n)",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "integer", "minimum": 0},
                                "description": "single [start, stop] inclusive interval"
                            }
                        },
                    ],
                    "items": {"type": "null"}
                }
            }
        },
        "required": ["relationships"]
    },
    "strict": True
}

#: Temporal-relationship JSON schema (verbatim): adds a relationship_type enum as a 5th element.
TEMPORAL_RELATION_SCHEMA: dict[str, Any] = {
    "name": "video_scene_graph_temporal",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "array",
                    "description": "[subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...], relationship_type]",
                    "minItems": 5,
                    "maxItems": 5,
                    "prefixItems": [
                        {"type": "integer", "description": "subject_id (may be -1 for camera)"},
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                            "description": "concise relation string, no underscores"
                        },
                        {"type": "integer", "description": "object_id (may be -1 for camera)"},
                        {
                            "type": "array",
                            "minItems": 1,
                            "description": "list of [start, end] inclusive intervals (frames 0..n)",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "integer", "minimum": 0},
                                "description": "single [start, stop] inclusive interval"
                            }
                        },
                        {
                            "type": "string",
                            "enum": ["functional", "stateful", "motion", "social", "attentional", "event_level"]
                        }
                    ],
                    "items": {"type": "null"}
                }
            }
        },
        "required": ["relationships"]
    },
    "strict": True
}

#: Temporal (non-spatial) relationship system prompt (verbatim, the "NOICL" variant).
TEMPORAL_RELATION_SYS_PROMPT = """## Role
You are a detail-oriented **Video Relationship Annotator** tasked with reviewing sequences of sampled video frames and extracting a comprehensive set of **temporal (non-spatial) relationships**. Ensure that all extracted relationships are visually grounded, type-consistent, and strictly adhere to the defined schema.

You are analyzing videos sampled at 1 fps. Each frame contains detected objects with corresponding bounding boxes. Please analyze all frames jointly and output temporal relationships according to the system instructions.

## Relationship Taxonomy
Classify each relationship into exactly ONE category:

1. **Functional — Contact / Manipulation**
   - Direct physical interaction where an animate subject alters or uses the state of another object.
   - Subject: animate | Object: animate or inanimate
   - Exclude: Pure motion or gaze without contact.

2. **Stateful — Attachment / Possession-like**
   - Visually grounded, time-persistent attachment or carrying relationships that indicate sustained physical association rather than instantaneous action. 
   - Subject: animate or inanimate | Object: animate or inanimate
   - Exclude: Abstract ownership or purely spatial layout.

3. **Motion — Relative Movement**
   - Temporal changes in relative position or movement trajectory between entities.
   - Subject: movable (animate or movable inanimate) | Object: animate or inanimate
   - Exclude: Static layout or manipulation actions.

4. **Social — Animate-to-Animate Interaction**
   - Communication, coordination, or interpersonal acts between animate agents.
   - Subject/Object: animate
   - Exclude: One-sided attention or non-social contact.

5. **Attentional — Gaze / Focus (includes Camera)**
   - Visual attention or camera focus directed at another object or agent.
   - Subject: animate or camera (with `object_id = -1`) | Object: animate or inanimate
   - Exclude: Communication or manipulation.
   - **For the camera (`object_id = -1`), only extract main or relevant relationships such as its movement or meaningful interaction with other objects; ignore obvious relationships such as "observing" or "recording" unless they are non-trivial or central to the event.**

6. **Event-Level — Goal-Directed Multi-Step Activity**
   - Higher-level, time-extended actions combining multiple functional, causality or motion relations into a single purposeful event. 
   - Subject: animate or inanimate | Object: animate or inanimate
   - Exclude: Single short actions or ungrounded intent.

## Core Analysis Logic & Constraints
1. **Object Typing:**
   - Animate: humans, animals, humanoid robots
   - Inanimate: cars, tools, furniture, etc.
   - Camera: unseen observer/recorder, always `object_id = -1`
2. **Typing Rules:**
   - Functional / Social: Subject must be animate
   - Motion: Subject must be movable
   - Attentional: Subject must be animate or camera (`-1`)
   - Social: Both subject and object must be animate
3. **Extraction Basis:**
   - All relationships must be visually supported and logically consistent with common sense. Do **NOT** infer relationships not visually evidenced or that contradict common sense.
   - Every object must be referenced by its bounding box coordinates `[x1, y1, x2, y2]` in each frame; all reasoning must rely on these spatial positions, object names and their temporal changes.
   - Relationships must have temporal grounding: define one or more time spans `[[start_frame, end_frame], ...]` as continuous frame intervals supported by visual evidence. If an object in a labeled relationship disappears from subsequent frames, end the relationship at the object's last visible frame. Do not continue relationships if objects become occluded or are missing.
   - Do NOT create self-relations (subject_id == object_id), e.g. (i, verb, i, ...).
   - If two object IDs correspond to the same physical entity, do not report relationships between them.
   - **NOTE: Do not ignore objects that newly appear in the middle of the video; evaluate all newly-detected objects for relationships as soon as they appear.**

## Input Format
The input is an ordered sequence of (image URL, text) pairs:
- Each text entry gives the frame ID, frame size (width and height), and detected objects with unique IDs, labels, and bounding boxes `[x1, y1, x2, y2]`.
- Frames are sampled at 1 fps and share consistent object IDs.

## Output Format
Produce a single valid JSON object using this schema:

```json
{
  "relationships": [
    [subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...], relationship_type]
  ]
}
```

### Output Details
- Each tuple must have five elements in this order:
subject_id (int) | predicate_verb (str) | object_id (int) | time_frames (list of [start,end]) | relationship_type (str ∈ {functional, stateful, motion, social, attentional, event_level})
- If there are no valid relationships, output `{ "relationships": [] }`.
- Do not emit error fields or any content outside the defined schema in your output.
- Always output strictly formatted, valid JSON with the "relationships" key.
"""

#: Spatial relationship system prompt (verbatim).
SPATIAL_RELATION_SYS_PROMPT = """
## Role
You are a detail-oriented **Video Relationship Annotator** responsible for reviewing sequences of sampled video frames and extracting a comprehensive set of **spatial relationships**. All relationships must be visually grounded, type-consistent, and strictly follow the defined schema.

Begin with a concise checklist (3–7 bullets) of what you will do; keep items conceptual, not implementation-level.

## Task Context
- Videos are sampled at 1 fps.
- Each frame contains detected objects with bounding boxes and corresponding unique integer IDs.
- Your analysis should consider all sampled frames jointly and produce relationships in accordance with the system instructions.

## Input Format
The input is an ordered sequence of (image URL, text) pairs:
- Each text entry includes:
  - Frame ID (integer index),
  - Frame size (width and height),
  - Detected objects (unique integer IDs, labels, and bounding boxes `[x1, y1, x2, y2]`).
- Frames maintain consistent object IDs when possible. If an object is missing in a frame, its ID is absent for that frame.

## Guidelines
- Objects are always referenced using their bounding boxes `[x1, y1, x2, y2]` in each frame; base all reasoning on these and their temporal evolution.
- Extract **only purely spatial (physical or geometric) relationships** visible in the 3D scene. Do **not** extract any relationships that indicate state, function, action, or purpose.
- **Exclude** all temporal, social, functional, or attentional relationships, as well as any stateful or action-based verbs.
- Each relationship must be visually supported by the frames.
- Do **NOT** ignore objects that newly appear in the middle of the video; ensure to evaluate all newly-detected objects for relationships as soon as they appear.
- Ensure logical consistency with common sense and real-world physics; do **NOT** output implausible or unsupported relationships.
- Think in terms of **3D spatial layout** by using depth information derived from world knowledge and visual cues, not just 2D image positions. Do **NOT** rely solely on 2D bounding box coordinates. Do **NOT** output `left of` or `right of`.
- Use precise, explicit, and non-redundant verbs.
- Do **NOT** miss any clear and valid spatial relationships between objects.


### Temporal Grounding
- Output relationships with one or more time spans (`[[start_frame, end_frame], ...]`), as continuous intervals where the relationship is visually supported and both objects are present.

## Output Format
Return a single valid JSON object following this schema:

```json
{
  "relationships": [
    [subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...]]
  ]
}
```

### Output Specifications
- Each relationship is a tuple with 4 elements, in order: subject_id (int) | predicate_verb (str) | object_id (int) | time_frames (list of [start,end]).
- If there are no valid relationships, output `{ "relationships": [] }`.
- Do not emit error fields or any content outside the defined schema in your output.
- Always output strictly formatted, valid JSON with the "relationships" key.

After producing the output, validate that all relationships are visually supported, ids and frames are valid integers, and the JSON format strictly matches the schema. If validation fails, self-correct before finalizing the output.
"""


def _encode_frame_data_url(frame_rgb: np.ndarray, quality: int = 90) -> str:
    """JPEG-encode an RGB frame as a base64 ``data:`` URL for the OpenAI vision API."""
    import base64
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _relationship_frame_indices(total_frames: int, fps: float, target_fps: float, max_frames: int) -> list[int]:
    """Original frame indices sampled at ~``target_fps``, capped (uniformly) at ``max_frames``."""
    step = max(1, round(fps / max(target_fps, 1e-6)))
    idxs = list(range(0, total_frames, step))
    if len(idxs) > max_frames:
        picks = sorted({int(p) for p in np.linspace(0, len(idxs) - 1, max_frames)})
        idxs = [idxs[i] for i in picks]
    return idxs


def _build_relationship_user_content(cfg: RelationshipConfig, frames: list[np.ndarray], sampled_indices: list[int],
                                     objects: list[dict[str, Any]], height: int, width: int) -> list[dict[str, Any]]:
    """Per sampled frame, an image item followed by a text block listing that frame's objects.

    Mirrors the reference ``build_chat_completion_line_new`` layout (image then text, raw-pixel
    bboxes, ``keep_uncertain`` filter), so the model sees the same input format.
    """
    content: list[dict[str, Any]] = []
    for frame_id, orig_idx in enumerate(sampled_indices):
        regions = []
        for obj in objects:
            box = obj["boxes"].get(str(orig_idx))
            if box is None:
                continue
            label = obj["name"]
            if (not cfg.keep_uncertain) and "uncertain" in label:
                continue
            regions.append({"id": obj["object_id"], "label": label, "bbox": box})
        text_item = {
            "frame id": frame_id,
            "frame width": width if regions else "",
            "frame height": height if regions else "",
            "objects": regions,
        }
        content.append({"type": "image_url",
                        "image_url": {"url": _encode_frame_data_url(frames[orig_idx]), "detail": cfg.image_detail}})
        content.append({"type": "text", "text": str(text_item)})
    return content


class RelationshipExtractor:
    """Extract temporal / spatial relationships, one OpenAI vision call per kind."""

    def __init__(self, cfg: RelationshipConfig) -> None:
        from openai import OpenAI

        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {cfg.api_key_env} is not set; it is required for stage 6.")
        self.cfg = cfg
        self.client = OpenAI(api_key=key, base_url=cfg.base_url)
        self._spec = {
            "temporal": (TEMPORAL_RELATION_SYS_PROMPT, TEMPORAL_RELATION_SCHEMA),
            "spatial": (SPATIAL_RELATION_SYS_PROMPT, SPATIAL_RELATION_SCHEMA),
        }

    def extract(self, user_content: list[dict[str, Any]], kind: str) -> list:
        """Return the ``relationships`` list for one kind (``"temporal"`` or ``"spatial"``)."""
        sys_prompt, schema = self._spec[kind]
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]
        extra: dict[str, Any] = {}
        if self.cfg.reasoning_effort:
            extra["reasoning_effort"] = self.cfg.reasoning_effort

        last_error: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model_id,
                    messages=messages,
                    max_completion_tokens=self.cfg.max_completion_tokens,
                    response_format={"type": "json_schema", "json_schema": schema},
                    **extra,
                )
                data = json.loads(response.choices[0].message.content)
                return data.get("relationships", [])
            except Exception as exc:  # noqa: BLE001 - retry on any API/parse error
                last_error = exc
                logger.warning("%s relationship attempt %d/%d failed: %s", kind, attempt + 1, self.cfg.max_retries, exc)
                extra.pop("reasoning_effort", None)  # some models reject it; drop and retry
        raise RuntimeError(f"Failed to extract {kind} relationships after {self.cfg.max_retries} attempts") from last_error


def stage6_relationships(cfg: PipelineConfig, scene_graph: dict[str, Any], frames: list[np.ndarray],
                         fps: float, extractor: RelationshipExtractor) -> dict[str, Any]:
    """Stage 6: add temporal + spatial relationships between the named objects.

    Consumes the stage-5 scene graph (named objects + per-frame boxes) and the video frames,
    samples frames at ~1 fps, and issues one vision call per relationship kind. Relationship
    tuples reference object ids; their ``[start, end]`` intervals index the sampled-frame
    sequence (see ``relationships.sampled_frame_indices`` for the mapping to original frames).
    """
    rcfg = cfg.relationship
    height, width = scene_graph["height"], scene_graph["width"]
    total_frames = scene_graph["total_frames"]
    objects = [
        {"object_id": o["object_id"], "name": o["name"], "boxes": o["trajectory"]["boxes"]}
        for o in scene_graph["objects"]
    ]
    sampled = _relationship_frame_indices(total_frames, fps, rcfg.frame_sample_fps, rcfg.max_frames)
    logger.info("Stage 6: %d named objects over %d frames sampled at ~%.2f fps",
                len(objects), len(sampled), rcfg.frame_sample_fps)
    user_content = _build_relationship_user_content(rcfg, frames, sampled, objects, height, width)

    relationships: dict[str, Any] = {"sampled_frame_indices": sampled, "temporal": [], "spatial": []}
    if rcfg.extract_temporal:
        relationships["temporal"] = extractor.extract(user_content, "temporal")
        logger.info("Stage 6: %d temporal relationships", len(relationships["temporal"]))
    if rcfg.extract_spatial:
        relationships["spatial"] = extractor.extract(user_content, "spatial")
        logger.info("Stage 6: %d spatial relationships", len(relationships["spatial"]))

    return {**scene_graph, "relationships": relationships}


# ======================================================================================
# Section 7 — Per-object trajectory helpers (stage 4/5)
# ======================================================================================


class ObjectTrajectories:
    """Decoded view over a stage-2/3 tracks artifact, indexed by object id."""

    def __init__(self, tracks_artifact: dict[str, Any]) -> None:
        self.height = tracks_artifact["height"]
        self.width = tracks_artifact["width"]
        self.total_frames = tracks_artifact["total_frames"]
        # active frames + decoded masks per object id
        self._active: dict[int, list[int]] = {}
        self._masks: dict[int, dict[int, np.ndarray]] = {}
        self._first_frame: dict[int, int] = {}
        for obj in tracks_artifact["objects"]:
            oid = obj["object_id"]
            self._first_frame[oid] = obj.get("first_frame", 0)
            active, decoded = [], {}
            for frame_idx, rle in enumerate(obj["masks"]):
                if rle and rle_area(rle) > 0:
                    active.append(frame_idx)
                    decoded[frame_idx] = decode_rle(rle)
            self._active[oid] = active
            self._masks[oid] = decoded

    def object_ids(self) -> list[int]:
        return list(self._active.keys())

    def total_area(self, obj_id: int) -> int:
        return int(sum(int(m.sum()) for m in self._masks[obj_id].values()))

    def largest_objects(self, k: int) -> list[int]:
        """Object ids with the largest total mask area (descending), keeping only non-empty ones."""
        scored = [(oid, self.total_area(oid)) for oid in self._active if self.total_area(oid) > 0]
        scored.sort(key=lambda x: -x[1])
        return [oid for oid, _ in scored[:k]]

    def sample_frames(self, obj_id: int, n: int, method: str = "uniform") -> list[int]:
        """Pick up to ``n`` frame indices where the object is visible."""
        active = self._active[obj_id]
        if not active:
            return []
        n = min(n, len(active))
        if method == "max_area":
            areas = [int(self._masks[obj_id][f].sum()) for f in active]
            top = np.argsort(areas)[-n:]
            picks = sorted(int(i) for i in top)
        else:  # uniform
            picks = np.linspace(0, len(active) - 1, n, dtype=int).tolist()
        return [active[i] for i in picks]

    def mask(self, obj_id: int, frame_idx: int) -> Optional[np.ndarray]:
        return self._masks[obj_id].get(frame_idx)

    def first_frame(self, obj_id: int) -> int:
        return self._first_frame.get(obj_id, 0)


# ======================================================================================
# Section 8 — Pipeline stages
# ======================================================================================


def stage1_generate_masks(cfg: PipelineConfig, frames: list[np.ndarray], mask_generator: AutomaticMaskGenerator,
                          fps: float, height: int, width: int) -> dict[str, Any]:
    """Stage 1: run automatic mask generation on sampled frames, keep the max-non-overlapping set."""
    rate = max(1, cfg.mask_gen.frame_sample_rate)
    sampled = list(range(0, len(frames), rate))
    logger.info("Stage 1: generating masks on %d / %d frames (every %d)", len(sampled), len(frames), rate)

    per_frame: list[list[dict]] = []
    for frame_idx in sampled:
        masks = mask_generator.generate(frames[frame_idx])
        kept = select_max_non_overlapping(masks, cfg.mask_gen.max_overlap_ratio)
        if not kept:
            per_frame.append([empty_rle(height, width)])
        else:
            per_frame.append([encode_rle(m) for m in kept])
        logger.info("  frame %d: %d -> %d masks", frame_idx, len(masks), len(kept))

    return {
        "video": cfg.io.video_path,
        "height": height,
        "width": width,
        "fps": fps,
        "total_frames": len(frames),
        "frame_sample_rate": rate,
        "sampled_frame_indices": sampled,
        "frames": per_frame,
    }


def stage2_track(cfg: PipelineConfig, masks_artifact: dict[str, Any], video_path: str,
                 tracker: VideoTracker) -> dict[str, Any]:
    """Stage 2: two-pass propagation with mid-video re-discovery of new objects.

    Pass 1 discovers objects and their first-appearance frames: it seeds from the first
    non-empty sampled frame, propagates, and whenever a large untracked region is covered by
    stage-1 detections it adds those as new objects and re-propagates from there. Pass 2
    resets, re-seeds every discovered object at its first-appearance frame, and propagates
    from frame 0 for clean, consistent per-object trajectories.
    """
    height, width = masks_artifact["height"], masks_artifact["width"]
    total_frames = masks_artifact["total_frames"]
    tcfg = cfg.tracking

    # stage-1 masks keyed by frame index
    sampled = masks_artifact["sampled_frame_indices"]
    stage1_masks: dict[int, list[np.ndarray]] = {}
    for frame_idx, rle_list in zip(sampled, masks_artifact["frames"]):
        stage1_masks[frame_idx] = [decode_rle(r) for r in rle_list if rle_area(r) > 0]

    # re-discovery check stride (scaled for long videos, as in the original)
    check_frames = set(sampled)
    if tcfg.adaptive_sample_rate:
        rate = masks_artifact["frame_sample_rate"] * (total_frames // 100 + 1)
        check_frames = {f for f in sampled if f % rate == 0}

    seed_frame = next((f for f in sampled if stage1_masks.get(f)), None)
    if seed_frame is None:
        logger.warning("Stage 2: no detections in any sampled frame; returning empty tracks")
        return _empty_tracks_artifact(cfg, height, width, total_frames)

    state = tracker.init_state(video_path)

    # ---- Pass 1: discovery -----------------------------------------------------------
    first_appearance: dict[int, tuple[int, np.ndarray]] = {}
    next_obj_id = 0
    for mask in stage1_masks[seed_frame]:
        tracker.add_mask(state, seed_frame, next_obj_id, mask)
        first_appearance[next_obj_id] = (seed_frame, mask)
        next_obj_id += 1
    logger.info("Stage 2 pass 1: seeded %d objects at frame %d", next_obj_id, seed_frame)

    frame_starter = seed_frame
    while frame_starter < total_frames:
        pending: Optional[tuple[int, list[np.ndarray]]] = None
        for frame_idx, tracked in tracker.propagate(state, frame_starter, total_frames - frame_starter - 1):
            if not (tcfg.rediscovery_enabled and frame_idx in check_frames
                    and frame_idx > seed_frame and next_obj_id < tcfg.max_objects):
                continue
            detections = stage1_masks.get(frame_idx, [])
            if not detections:
                continue
            new_objs = _find_new_objects(tracked, detections, height, width, tcfg)
            if new_objs:
                pending = (frame_idx, new_objs)
                break  # stop propagation, add the new objects, then re-propagate from here
        if pending is None:
            break
        frame_idx, new_objs = pending
        new_objs = new_objs[: tcfg.max_objects - next_obj_id]
        for mask in new_objs:
            tracker.add_mask(state, frame_idx, next_obj_id, mask)
            first_appearance[next_obj_id] = (frame_idx, mask)
            next_obj_id += 1
        logger.info("Stage 2 pass 1: +%d objects at frame %d (total %d)", len(new_objs), frame_idx, next_obj_id)
        frame_starter = frame_idx + 1

    logger.info("Stage 2 pass 1 complete: %d objects discovered", next_obj_id)

    # ---- Pass 2: clean collection ----------------------------------------------------
    tracker.reset(state)
    for obj_id, (appear_frame, mask) in first_appearance.items():
        tracker.add_mask(state, appear_frame, obj_id, mask)
    trajectories: dict[int, list[Optional[dict]]] = {oid: [None] * total_frames for oid in range(next_obj_id)}
    for frame_idx, tracked in tracker.propagate(state, 0, total_frames - 1):
        for obj_id, mask in tracked.items():
            if obj_id < next_obj_id and mask.any():
                trajectories[obj_id][frame_idx] = encode_rle(mask)

    objects = [
        {"object_id": oid, "first_frame": first_appearance[oid][0], "masks": trajectories[oid]}
        for oid in range(next_obj_id)
    ]
    return {
        "video": cfg.io.video_path,
        "height": height,
        "width": width,
        "total_frames": total_frames,
        "objects": objects,
    }


def _find_new_objects(tracked: dict[int, np.ndarray], detections: list[np.ndarray], height: int, width: int,
                      tcfg: TrackingConfig) -> list[np.ndarray]:
    """Return stage-1 detections that should be added as new objects this frame."""
    if tracked:
        union = np.zeros((height, width), dtype=bool)
        for m in tracked.values():
            union |= m
    else:
        union = np.zeros((height, width), dtype=bool)
    untracked = ~union
    untracked_area = int(untracked.sum())
    if untracked_area / float(height * width) < tcfg.rediscovery_min_untracked_frac:
        return []

    tracked_ids = list(tracked.keys())
    tracked_masks = list(tracked.values())
    assigned = match_detections_to_tracks(tracked_ids, tracked_masks, detections, tcfg.match_iou_thresh)
    existing = set(tracked_ids)

    new_objs: list[np.ndarray] = []
    for det, det_id in zip(detections, assigned):
        if det_id in existing:
            continue  # this detection corresponds to an already-tracked object
        overlap = int(np.logical_and(det, untracked).sum())
        if untracked_area > 0 and overlap / float(untracked_area) >= tcfg.rediscovery_overlap_thresh:
            new_objs.append(det)
    return new_objs


def _empty_tracks_artifact(cfg: PipelineConfig, height: int, width: int, total_frames: int) -> dict[str, Any]:
    return {
        "video": cfg.io.video_path,
        "height": height,
        "width": width,
        "total_frames": total_frames,
        "objects": [],
    }


def stage3_cleanup(cfg: PipelineConfig, tracks_artifact: dict[str, Any]) -> dict[str, Any]:
    """Stage 3: de-duplicate contained tracks and morphologically clean each mask."""
    ccfg = cfg.cleanup
    objects = tracks_artifact["objects"]
    if not ccfg.enabled or not objects:
        return tracks_artifact

    tracks = [obj["masks"] for obj in objects]
    alive = dedupe_tracks(tracks, ccfg.contain_ratio_thresh, ccfg.coverage_share_thresh, ccfg.min_overlap_frames)
    logger.info("Stage 3: %d -> %d tracks after de-duplication", len(objects), sum(alive))

    cleaned: list[dict[str, Any]] = []
    for obj, keep in zip(objects, alive):
        if not keep:
            continue
        new_masks: list[Optional[dict]] = []
        for rle in obj["masks"]:
            if not rle or rle_area(rle) == 0:
                new_masks.append(None)
                continue
            opened = morphological_open(decode_rle(rle), ccfg.morph_kernel, ccfg.morph_iterations)
            new_masks.append(encode_rle(opened) if int(opened.sum()) >= ccfg.morph_min_area else None)
        if any(m is not None for m in new_masks):
            cleaned.append({"object_id": obj["object_id"], "first_frame": obj["first_frame"], "masks": new_masks})

    return {**{k: tracks_artifact[k] for k in ("video", "height", "width", "total_frames")}, "objects": cleaned}


def stage4_caption(cfg: PipelineConfig, tracks_artifact: dict[str, Any], frames: list[np.ndarray],
                   captioner: DamCaptioner) -> dict[str, Any]:
    """Stage 4: describe the largest objects with the region captioner."""
    trajectories = ObjectTrajectories(tracks_artifact)
    targets = trajectories.largest_objects(cfg.caption.max_objects)
    logger.info("Stage 4: describing %d objects", len(targets))

    described = []
    for obj_id in targets:
        sampled_frames = trajectories.sample_frames(obj_id, cfg.caption.frames_per_object, cfg.caption.frame_sampling)
        if not sampled_frames:
            continue
        frame_imgs = [frames[f] for f in sampled_frames]
        masks = [trajectories.mask(obj_id, f) for f in sampled_frames]
        description = captioner.describe(frame_imgs, masks)
        logger.info("  object %d: %s", obj_id, description[:80].replace("\n", " "))
        described.append({"object_id": obj_id, "description": description, "sampled_frames": sampled_frames})

    return {"video": cfg.io.video_path, "objects": described}


def stage5_structure(cfg: PipelineConfig, descriptions_artifact: dict[str, Any], tracks_artifact: dict[str, Any],
                     structurer: SceneGraphStructurer) -> dict[str, Any]:
    """Stage 5: turn each description into a structured record and assemble the final scene graph."""
    trajectories = ObjectTrajectories(tracks_artifact)
    masks_by_object = {obj["object_id"]: obj["masks"] for obj in tracks_artifact["objects"]}

    objects = []
    for entry in descriptions_artifact["objects"]:
        obj_id = entry["object_id"]
        record = structurer.structure(entry["description"])
        masks = masks_by_object.get(obj_id, [])
        boxes = {}
        active_frames = []
        for frame_idx, rle in enumerate(masks):
            if rle and rle_area(rle) > 0:
                active_frames.append(frame_idx)
                box = mask_bbox(decode_rle(rle))
                if box is not None:
                    boxes[str(frame_idx)] = box
        objects.append({
            "object_id": obj_id,
            "name": record["Object"],
            "attributes": record["Attributes"],
            "relationships": record["Relationships"],
            "actions": record["Actions"],
            "description": entry["description"],
            "trajectory": {
                "first_frame": trajectories.first_frame(obj_id),
                "frames": active_frames,
                "boxes": boxes,
                "masks": masks,
            },
        })
        logger.info("  object %d -> %s", obj_id, record["Object"])

    return {
        "video": cfg.io.video_path,
        "height": tracks_artifact["height"],
        "width": tracks_artifact["width"],
        "total_frames": tracks_artifact["total_frames"],
        "objects": objects,
    }


# ======================================================================================
# Section 9 — Orchestrator
# ======================================================================================


class Pipeline:
    """Runs the configured range of stages, lazily loading only the models it needs."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg.normalize()
        os.environ.setdefault("HF_HOME", str(Path(cfg.io.cache_dir) / "hf"))
        _add_local_packages(cfg.io.cache_dir)
        self._frames: Optional[list[np.ndarray]] = None
        self._video_meta: Optional[tuple[float, int, int]] = None

    # ---- lazy resources -------------------------------------------------------------

    def _ensure_frames(self) -> tuple[list[np.ndarray], float, int, int]:
        if self._frames is None:
            frames, fps, height, width = read_video_frames(self.cfg.io.video_path)
            self._frames = frames
            self._video_meta = (fps, height, width)
        fps, height, width = self._video_meta  # type: ignore[misc]
        return self._frames, fps, height, width

    # ---- artifact persistence -------------------------------------------------------

    def _load_artifact(self, stage_idx: int) -> dict[str, Any]:
        path = artifact_path(self.cfg, stage_idx)
        if not path.exists():
            raise FileNotFoundError(
                f"Stage {stage_idx} artifact not found at {path}. Run stages <= {stage_idx} first, "
                f"or lower --start-stage."
            )
        logger.info("Loading stage-%d artifact from %s", stage_idx, path)
        return read_json(path)

    def _save_artifact(self, stage_idx: int, data: dict[str, Any]) -> None:
        if self.cfg.should_save(stage_idx):
            path = artifact_path(self.cfg, stage_idx)
            write_json(path, data)
            logger.info("Wrote stage-%d artifact to %s", stage_idx, path)

    # ---- stage runners --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute ``start_stage..end_stage`` and return the final stage's artifact."""
        cfg = self.cfg
        logger.info("Running stages %d..%d", cfg.start_stage, cfg.end_stage)

        artifact: dict[str, Any] = {}
        if cfg.start_stage > 1:
            artifact = self._load_artifact(cfg.start_stage - 1)

        for stage_idx in range(cfg.start_stage, cfg.end_stage + 1):
            artifact = self._run_stage(stage_idx, artifact)
            self._save_artifact(stage_idx, artifact)
        return artifact

    def _run_stage(self, stage_idx: int, prev: dict[str, Any]) -> dict[str, Any]:
        name = STAGE_NAMES[stage_idx - 1]
        logger.info("=== Stage %d (%s) ===", stage_idx, name)
        if stage_idx == 1:
            frames, fps, height, width = self._ensure_frames()
            # Stage 1 always uses SAM 2 (independent of the tracking backbone).
            amg = AutomaticMaskGenerator(self.cfg.mask_gen, self.cfg.runtime.device)
            return stage1_generate_masks(self.cfg, frames, amg, fps, height, width)
        if stage_idx == 2:
            # Official SAM 2 video predictor reads the video file directly (no decoded frames needed).
            tracker = VideoTracker(self.cfg.tracking.model_id, self.cfg.runtime.device,
                                   self.cfg.tracking.offload_to_cpu)
            return stage2_track(self.cfg, prev, self.cfg.io.video_path, tracker)
        if stage_idx == 3:
            return stage3_cleanup(self.cfg, prev)
        if stage_idx == 4:
            frames, _, _, _ = self._ensure_frames()
            captioner = DamCaptioner(self.cfg.caption, self.cfg.runtime.device)
            return stage4_caption(self.cfg, prev, frames, captioner)
        if stage_idx == 5:
            tracks = self._load_artifact(3)  # final output needs the cleaned trajectories
            structurer = SceneGraphStructurer(self.cfg.structure)
            return stage5_structure(self.cfg, prev, tracks, structurer)
        if stage_idx == 6:
            frames, fps, _, _ = self._ensure_frames()  # relationship reasoning needs the frame images
            extractor = RelationshipExtractor(self.cfg.relationship)
            return stage6_relationships(self.cfg, prev, frames, fps, extractor)
        raise AssertionError(stage_idx)


# ======================================================================================
# Section 10 — CLI
# ======================================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SVG2 — Video Scene Graph pipeline (SAM 2 + DAM + OpenAI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON config file.")
    parser.add_argument("--dump-config", type=str, default=None,
                        help="Write the effective config to this path and exit.")

    # IO
    parser.add_argument("--video", dest="video_path", type=str, help="Input video path.")
    parser.add_argument("--output-dir", type=str, help="Artifact output directory.")
    parser.add_argument("--cache-dir", type=str, help="Model / HF cache directory.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if artifacts exist.")

    # Runtime
    parser.add_argument("--device", type=str, help='Torch device (e.g. "cuda", "cpu").')
    parser.add_argument("--log-level", type=str, help="Logging level (INFO, DEBUG, ...).")

    # Stage control
    parser.add_argument("--start-stage", type=str, help="First stage to run (name or 1-5).")
    parser.add_argument("--end-stage", type=str, help="Last stage to run (name or 1-5).")
    parser.add_argument("--save-stages", type=str,
                        help="Comma-separated stages whose artifacts to save (e.g. '1,2,3').")
    parser.add_argument("--no-save-intermediate", action="store_true",
                        help="Only save the final (end-stage) artifact.")

    # SAM 2 models (stage 1 mask generation + stage 2 tracking)
    parser.add_argument("--maskgen-model-id", type=str, help="Stage 1: SAM 2 AMG HF repo id.")
    parser.add_argument("--tracking-model-id", type=str, help="Stage 2: SAM 2 video predictor HF repo id.")

    # Common per-stage knobs (full set lives in the config file)
    parser.add_argument("--frame-sample-rate", type=int, help="Stage 1: sample every Nth frame.")
    parser.add_argument("--max-objects", type=int, help="Stage 2: cap on tracked objects.")
    parser.add_argument("--caption-model-id", type=str, help="Stage 4: DAM model id.")
    parser.add_argument("--structure-model-id", type=str, help="Stage 5: OpenAI model id.")
    parser.add_argument("--relationship-model-id", type=str, help="Stage 6: OpenAI (vision) model id.")
    parser.add_argument("--relationship-fps", type=float, help="Stage 6: frame sampling rate (fps).")
    parser.add_argument("--relationship-max-frames", type=int, help="Stage 6: max frames per API call.")
    parser.add_argument("--no-temporal", action="store_true", help="Stage 6: skip temporal relationships.")
    parser.add_argument("--no-spatial", action="store_true", help="Stage 6: skip spatial relationships.")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a base file (optional) + CLI overrides."""
    cfg = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()

    # IO
    if args.video_path is not None:
        cfg.io.video_path = args.video_path
    if args.output_dir is not None:
        cfg.io.output_dir = args.output_dir
    if args.cache_dir is not None:
        cfg.io.cache_dir = args.cache_dir
    if args.overwrite:
        cfg.io.overwrite = True

    # Runtime
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.log_level is not None:
        cfg.runtime.log_level = args.log_level

    # Stage control
    if args.start_stage is not None:
        cfg.start_stage = args.start_stage  # normalized later
    if args.end_stage is not None:
        cfg.end_stage = args.end_stage
    if args.save_stages is not None:
        cfg.save_stages = [int(s) for s in args.save_stages.split(",") if s.strip()]
    if args.no_save_intermediate:
        cfg.save_stages = []

    # SAM 2 model ids
    if args.maskgen_model_id is not None:
        cfg.mask_gen.model_id = args.maskgen_model_id
    if args.tracking_model_id is not None:
        cfg.tracking.model_id = args.tracking_model_id

    # Per-stage knobs
    if args.frame_sample_rate is not None:
        cfg.mask_gen.frame_sample_rate = args.frame_sample_rate
    if args.max_objects is not None:
        cfg.tracking.max_objects = args.max_objects
    if args.caption_model_id is not None:
        cfg.caption.model_id = args.caption_model_id
    if args.structure_model_id is not None:
        cfg.structure.model_id = args.structure_model_id
    if args.relationship_model_id is not None:
        cfg.relationship.model_id = args.relationship_model_id
    if args.relationship_fps is not None:
        cfg.relationship.frame_sample_fps = args.relationship_fps
    if args.relationship_max_frames is not None:
        cfg.relationship.max_frames = args.relationship_max_frames
    if args.no_temporal:
        cfg.relationship.extract_temporal = False
    if args.no_spatial:
        cfg.relationship.extract_spatial = False

    return cfg.normalize()


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)

    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.dump_config:
        cfg.to_file(args.dump_config)
        logger.info("Wrote effective config to %s", args.dump_config)
        return

    if cfg.start_stage == 1 and not cfg.io.video_path:
        raise SystemExit("--video is required when starting from stage 1.")

    final = Pipeline(cfg).run()
    logger.info("Done. Final artifact has %d objects.", len(final.get("objects", [])))


if __name__ == "__main__":
    main()
