"""Train TRASER-F (scene graph forecasting) starting from the released TRASER weights.

    deepspeed traser_train/forecast/train_forecast.py --model_name_or_path UWGZQ/TRASER \
        --train_annotation data/psg4d_hoi_train.json --output_dir checkpoints/traser_f ...

See traser/scripts/train_forecast.sh for the full recipe (two stages).
"""

import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from traser_train.forecast.data_forecast import ForecastCollator, ForecastDataset
from traser_train.forecast.modeling_forecast import FORECASTER_DEFAULTS, TRASERForecast, init_forecaster
from traser_train.forecast.trainer_forecast import ForecastTrainer
from traser_train.train.argument import TrainingArguments
from traser_train.train.train_qwen import OBJ_TRAJ_TOKENS, add_obj_traj_tokens, get_language_model, get_visual


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="UWGZQ/TRASER", metadata={"help": "TRASER or TRASER-F checkpoint."})
    processor_name_or_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    reinit_forecaster: bool = field(default=True, metadata={
        "help": "Initialize the forecaster from scratch (set False when resuming from a TRASER-F checkpoint)."})
    forecaster_d_model: int = field(default=FORECASTER_DEFAULTS["forecaster_d_model"])
    forecaster_layers: int = field(default=FORECASTER_DEFAULTS["forecaster_layers"])
    forecaster_decoder_layers: int = field(default=FORECASTER_DEFAULTS["forecaster_decoder_layers"])
    forecaster_heads: int = field(default=FORECASTER_DEFAULTS["forecaster_heads"])
    forecaster_dropout: float = field(default=FORECASTER_DEFAULTS["forecaster_dropout"])


@dataclass
class DataArguments:
    train_annotation: str = field(default="")
    history_min_sec: int = field(default=1)
    history_max_sec: int = field(default=5)
    horizon_min_sec: int = field(default=1)
    horizon_max_sec: int = field(default=5)
    event_anchor_prob: float = field(default=0.7, metadata={
        "help": "Probability of drawing an anchor whose future contains a relationship start."})
    keep_empty_prob: float = field(default=0.1, metadata={"help": "Keep clips without future relationships."})
    object_filter: str = field(default="related", metadata={"help": "related | all"})
    max_objects: int = field(default=20)
    object_names_in_prompt: bool = field(default=True, metadata={
        "help": "Give object categories in the block labels (GAGS-like setting)."})
    video_max_frame_pixels: int = field(default=768 * 28 * 28)
    video_min_frame_pixels: int = field(default=64 * 28 * 28)
    # Evaluation-mode fields, unused in training.
    history_sec: int = field(default=3)
    horizon_sec: int = field(default=3)
    eval_stride: int = field(default=1)


@dataclass
class ForecastTrainingArguments(TrainingArguments):
    sample_fps: int = field(default=2, metadata={"help": "Sampled frames per second (2 -> 1 temporal grid / s)."})
    window_sec: int = field(default=1, metadata={"help": "Temporal window length in seconds (history and forecast)."})
    forecast_mode: str = field(default="latent", metadata={"help": "latent | copy | oracle | prompt"})
    lambda_latent: float = field(default=1.0)
    lambda_box: float = field(default=1.0)
    forecaster_lr: Optional[float] = field(default=1e-4)
    tune_forecaster: bool = field(default=True)


def set_trainable(model, margs, targs):
    for p in get_visual(model).parameters():
        p.requires_grad = margs.tune_mm_vision
    for p in get_visual(model).merger.parameters():
        p.requires_grad = margs.tune_mm_mlp
    for p in get_language_model(model).parameters():
        p.requires_grad = margs.tune_mm_llm
    model.lm_head.requires_grad_(margs.tune_mm_llm)
    for name in ("perceiver_resampler", "second_perceiver_resampler"):
        if hasattr(model, name):
            getattr(model, name).requires_grad_(targs.tune_mm_resampler)
    model.forecaster.requires_grad_(targs.tune_forecaster)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train / 1e6:.1f}M")


def train(attn_implementation="flash_attention_2"):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, ForecastTrainingArguments))
    margs, dargs, targs = parser.parse_args_into_dataclasses()
    targs.remove_unused_columns = False
    os.makedirs(targs.output_dir, exist_ok=True)

    model = TRASERForecast.from_pretrained(
        margs.model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if targs.bf16 else None),
        forecaster_d_model=margs.forecaster_d_model,
        forecaster_layers=margs.forecaster_layers,
        forecaster_decoder_layers=margs.forecaster_decoder_layers,
        forecaster_heads=margs.forecaster_heads,
        forecaster_dropout=margs.forecaster_dropout,
    )
    if margs.reinit_forecaster:
        init_forecaster(model)
    model.config.use_cache = False
    if targs.gradient_checkpointing:
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(margs.processor_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(margs.model_name_or_path, padding_side="right",
                                                           use_fast=False)

    class _D:  # add_obj_traj_tokens only needs .processor
        pass
    holder = _D()
    holder.processor = processor
    add_obj_traj_tokens(model, tokenizer, holder)
    model.config.obj_traj_start_id = tokenizer.convert_tokens_to_ids(OBJ_TRAJ_TOKENS[0])
    model.config.obj_traj_end_id = tokenizer.convert_tokens_to_ids(OBJ_TRAJ_TOKENS[1])

    set_trainable(model, margs, targs)
    train_set = ForecastDataset(dargs.train_annotation, processor, tokenizer,
                                _merge(dargs, targs), train=True)
    trainer = ForecastTrainer(model=model, processing_class=tokenizer, args=targs, train_dataset=train_set,
                              data_collator=ForecastCollator(pad_token_id=tokenizer.pad_token_id))

    resume = bool(list(pathlib.Path(targs.output_dir).glob("checkpoint-*")))
    if resume:
        logging.info("checkpoint found, resume training")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()
    processor.save_pretrained(targs.output_dir)
    model.config.use_cache = True
    if trainer.deepspeed:
        torch.cuda.synchronize()
    trainer.save_model(targs.output_dir)


def _merge(dargs, targs):
    """The dataset reads sample_fps / window_sec from the training arguments."""
    class _A:
        pass
    a = _A()
    a.__dict__.update(vars(dargs))
    a.sample_fps, a.window_sec = targs.sample_fps, targs.window_sec
    return a


if __name__ == "__main__":
    train()
