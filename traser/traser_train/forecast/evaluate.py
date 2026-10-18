"""Evaluate a TRASER-F checkpoint: generate the future relationships of every test clip.

    python -m traser_train.forecast.evaluate --model checkpoints/traser_f \
        --annotation data/psg4d_hoi_test.json --history_sec 3 --horizon_sec 3 --mode latent

Outside ``--mode oracle`` the model only receives the observed frames (the clip is cut at
the anchor before encoding), so nothing from the future can reach it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from traser_train.forecast.data_forecast import ForecastCollator, ForecastDataset
from traser_train.forecast.metrics import ForecastMeter, parse_relationships
from traser_train.forecast.modeling_forecast import TRASERForecast
from traser_train.forecast.step import TEMPORAL_PATCH_SIZE, build_forecast_inputs


def history_only(batch, frames_per_window: int):
    """Drop the future grids from the video and the masks (patches are ordered time-major)."""
    T, Hp, Wp = (int(x) for x in batch["video_grid_thw"][0])
    Th = int(batch["n_hist_windows"][0]) * frames_per_window // TEMPORAL_PATCH_SIZE
    batch["pixel_values_videos"] = batch["pixel_values_videos"][: Th * Hp * Wp]
    batch["video_grid_thw"] = torch.tensor([[Th, Hp, Wp]])
    batch["obj_masks"] = [m[:, : Th * TEMPORAL_PATCH_SIZE] for m in batch["obj_masks"]]
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--mode", default="latent", choices=["latent", "copy", "oracle", "prompt"])
    ap.add_argument("--history_sec", type=int, default=3)
    ap.add_argument("--horizon_sec", type=int, default=3)
    ap.add_argument("--eval_stride", type=int, default=1)
    ap.add_argument("--sample_fps", type=int, default=2)
    ap.add_argument("--window_sec", type=int, default=1)
    ap.add_argument("--object_filter", default="related")
    ap.add_argument("--max_objects", type=int, default=20)
    ap.add_argument("--object_names_in_prompt", type=int, default=1)
    ap.add_argument("--video_max_frame_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--video_min_frame_pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--coverage_thresh", type=float, default=0.5)
    ap.add_argument("--time_reduce", default="max")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N clips.")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--output", default=None, help="Write per-clip predictions (jsonl) here.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TRASERForecast.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                           attn_implementation=args.attn).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.processor)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    data = ForecastDataset(args.annotation, processor, tokenizer, args, train=False)
    collate = ForecastCollator(pad_token_id=tokenizer.pad_token_id)
    meter = ForecastMeter(args.horizon_sec)
    frames_per_window = args.sample_fps * args.window_sec
    out_f = open(args.output, "w") if args.output else None

    n = len(data) if not args.limit else min(args.limit, len(data))
    for i in range(n):
        sample = data[i]
        sample["input_ids"] = data.generation_prompt_ids(sample)
        sample["labels"] = None
        batch = collate([sample])
        if args.mode != "oracle":
            batch = history_only(batch, frames_per_window)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(device)
        batch["video_grid_thw"] = batch["video_grid_thw"].to(device)
        with torch.no_grad():
            emb, pid, mask, rope, _, _, aux = build_forecast_inputs(
                model, tokenizer, batch, sample_fps=args.sample_fps, window_sec=args.window_sec,
                coverage_thresh=args.coverage_thresh, time_reduce=args.time_reduce, mode=args.mode,
                with_teacher=(args.mode == "oracle"))
            gen = model.generate(inputs_embeds=emb, position_ids=pid, attention_mask=mask.long(),
                                 rope_deltas=rope, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = parse_relationships(text)
        meta = sample["meta"]
        a = aux[0]
        meter.add(pred, meta["target"], meta["history"], meta["horizon"],
                  pred_boxes=a["future_boxes"].cpu(), gt_boxes=sample["fut_boxes"],
                  gt_valid=sample["fut_box_valid"], vis_logits=a["vis_logits"].cpu(), window_sec=args.window_sec)
        if out_f:
            out_f.write(json.dumps(dict(video=meta["video"], t_sec=meta["t_sec"], objects=meta["obj_names"],
                                        text=text, pred=pred, target=meta["target"],
                                        boxes=a["future_boxes"].cpu().tolist())) + "\n")
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{n}] {meter.summary()}", file=sys.stderr, flush=True)

    print(json.dumps({"mode": args.mode, **meter.summary()}, indent=2))
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
