from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=128)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: int = field(
        default=1,
        metadata={"help": "Target interval between sampled video frames, in seconds."},
    )
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=768 * 28 * 28)
    video_min_frame_pixels: int = field(default=64 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=128000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    # Per-module learning rates.
    mm_projector_lr: Optional[float] = None
    resampler_lr: Optional[float] = None
    tune_mm_resampler: bool = field(default=True)

    # Trajectory-aware token selection.
    coverage_thresh: Optional[float] = field(
        default=0.5,
        metadata={"help": "Minimum object-mask coverage for a visual token to be selected."},
    )
    time_reduce: Optional[str] = field(
        default="max",
        metadata={"help": "Reduction of mask coverage across the frames merged into one grid: mean / max / min."},
    )

    # Trajectory-aligned token arrangement.
    temporal_window_length: int = field(
        default=4,
        metadata={"help": "Length of one temporal window, in seconds."},
    )
    training_fps: int = field(
        default=1,
        metadata={"help": "Frame sampling rate of the data pipeline (1 / DataArguments.base_interval)."},
    )
    # Token ids of <obj_traj_start> / <obj_traj_end>; filled in at runtime after the
    # special tokens have been added to the tokenizer.
    obj_traj_start_id: Optional[int] = field(default=None)
    obj_traj_end_id: Optional[int] = field(default=None)

    # Perceiver resamplers. Field names follow the released model config:
    #   - Temporal-Window Resampler (TWR): compresses one object's tokens inside each
    #     temporal window into `temporal_resampler_n_latents` latents.
    #   - Object-Trajectory Resampler (OTR): compresses all of one object's tokens across
    #     the whole video into `object_resampler_n_latents` latents.
    use_resampler: bool = field(default=True)
    object_resampler: bool = field(default=True)
    resampler_depth: int = field(default=3)
    temporal_resampler_n_latents: int = field(default=32)
    object_resampler_n_latents: int = field(default=32)
