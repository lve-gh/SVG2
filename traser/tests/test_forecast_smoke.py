"""CPU smoke test of the forecasting stack with a tiny random TRASER-F.

Builds a synthetic 8-second clip (a "hand" square moving onto a "bottle" square), then
checks, end to end:

1. dataset + collator produce a clip with H observed and F future windows;
2. history features do not depend on the future frames (no leakage through the encoder);
3. a freshly initialized forecaster equals the copy-last baseline;
4. two optimizer steps of ForecastTrainer run, every loss term is finite, and the
   forecaster receives gradients from the LM loss;
5. evaluation (history-only input, generation, parsing, metrics) and the baselines run.

    python traser/tests/test_forecast_smoke.py

Needs the Qwen2.5-VL processor/tokenizer files from the Hub (a few MB), no weights, no GPU.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycocotools import mask as maskUtils  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLConfig  # noqa: E402

# data_qwen imports decord at module level; without it, stub it and let frames fall back to OpenCV.
if importlib.util.find_spec("decord") is None:
    import types

    class _NoDecord:
        def __init__(self, *a, **k):
            raise ImportError("decord is not installed")

    sys.modules["decord"] = types.SimpleNamespace(VideoReader=_NoDecord)

from traser_train.forecast.baselines import MarkovBaseline, box_baseline, copy_last_relations  # noqa: E402
from traser_train.forecast.data_forecast import ClipSpec, ForecastCollator, ForecastDataset, clip_targets  # noqa: E402
from traser_train.forecast.evaluate import history_only  # noqa: E402
from traser_train.forecast.metrics import ForecastMeter, parse_relationships  # noqa: E402
from traser_train.forecast.modeling_forecast import TRASERForecast, init_forecaster  # noqa: E402
from traser_train.train.modeling_traser import TRASER  # noqa: E402
from traser_train.forecast.step import build_forecast_inputs  # noqa: E402
from traser_train.forecast.train_forecast import ForecastTrainingArguments  # noqa: E402
from traser_train.forecast.trainer_forecast import ForecastTrainer  # noqa: E402
from traser_train.train.train_qwen import OBJ_TRAJ_TOKENS, add_obj_traj_tokens  # noqa: E402

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
FPS, SECONDS, H_IMG, W_IMG = 15, 8, 224, 336


def make_clip(root: str):
    import cv2
    n = FPS * SECONDS
    video = os.path.join(root, "clip.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_IMG, H_IMG))
    masks = []
    for t in range(n):
        img = np.full((H_IMG, W_IMG, 3), 40, np.uint8)
        hand = np.zeros((H_IMG, W_IMG), np.uint8)
        bottle = np.zeros((H_IMG, W_IMG), np.uint8)
        bottle[120:190, 220:260] = 1
        x = int(20 + min(t, 90) * 2.0)
        hand[100:150, x:x + 50] = 1
        img[bottle > 0] = (30, 160, 230)
        img[hand > 0] = (230, 180, 140)
        vw.write(img[:, :, ::-1])
        masks.append([{"size": [H_IMG, W_IMG],
                       "counts": maskUtils.encode(np.asfortranarray(m))["counts"].decode()} for m in (hand, bottle)])
    vw.release()
    mask_json = os.path.join(root, "clip_rle.json")
    json.dump(masks, open(mask_json, "w"))
    ann = [dict(video=video, mask_json=mask_json, fps=FPS, num_frames=n,
                objects=[{"id": 0, "name": "hand"}, {"id": 1, "name": "bottle"}],
                relationships=[[0, "reachout", 1, [[15, 59]]], [0, "grasp", 1, [[60, 74]]],
                               [0, "pickup", 1, [[75, 104]]]])]
    path = os.path.join(root, "ann.json")
    json.dump(ann, open(path, "w"))
    return path


def tiny_model(tokenizer):
    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, vocab_size=len(tokenizer),
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64,
                           fullatt_block_indexes=[1]),
    )
    cfg.hidden_size = 64
    cfg.vision_start_token_id, cfg.vision_end_token_id = 151652, 151653
    for k, v in dict(temporal_resampler_n_latents=4, object_resampler_n_latents=4, resampler_depth=1,
                     object_resampler=True, forecaster_d_model=32, forecaster_layers=1,
                     forecaster_decoder_layers=1, forecaster_heads=4, forecaster_dropout=0.0).items():
        setattr(cfg, k, v)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = TRASERForecast(cfg)
    model.forecaster.reset_parameters()
    return model


def main():
    torch.manual_seed(0)
    root = tempfile.mkdtemp(prefix="traser_f_")
    ann = make_clip(root)
    processor = AutoProcessor.from_pretrained(BASE)
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    model = tiny_model(tokenizer)
    add_obj_traj_tokens(model, tokenizer, SimpleNamespace(processor=processor))
    model.config.obj_traj_start_id = tokenizer.convert_tokens_to_ids(OBJ_TRAJ_TOKENS[0])
    model.config.obj_traj_end_id = tokenizer.convert_tokens_to_ids(OBJ_TRAJ_TOKENS[1])

    dargs = SimpleNamespace(
        sample_fps=2, window_sec=1, history_min_sec=3, history_max_sec=3, horizon_min_sec=2, horizon_max_sec=2,
        event_anchor_prob=1.0, keep_empty_prob=1.0, object_filter="related", max_objects=20,
        object_names_in_prompt=True, video_max_frame_pixels=224 * 336, video_min_frame_pixels=56 * 56,
        history_sec=3, horizon_sec=2, eval_stride=1)

    # 1. dataset
    train_set = ForecastDataset(ann, processor, tokenizer, dargs, train=True)
    sample = train_set.build(ClipSpec(0, 3, 3, 2))
    assert sample["n_hist_windows"] == 3 and sample["n_fut_windows"] == 2
    assert sample["obj_masks"].shape[:2] == (2, 10), sample["obj_masks"].shape
    assert sample["meta"]["target"] == [[1, "reachout", 2, [[3, 3]]], [1, "grasp", 2, [[4, 4]]]], sample["meta"]["target"]
    print("1. dataset ok:", sample["meta"]["target"], "grid", sample["video_grid_thw"].tolist())

    # 2. no leakage: history features are the same with and without the future frames
    batch = ForecastCollator(tokenizer.pad_token_id)([sample])
    with torch.no_grad():
        full = model.model.get_video_features(batch["pixel_values_videos"], batch["video_grid_thw"])[0]
        hb = history_only(dict(batch), frames_per_window=2)
        hist = model.model.get_video_features(hb["pixel_values_videos"], hb["video_grid_thw"])[0]
    assert torch.allclose(full[: hist.shape[0]], hist, atol=1e-5), (full[: hist.shape[0]] - hist).abs().max()
    print("2. no leakage through the vision encoder ok")

    # 3. fresh forecaster == copy-last baseline
    with torch.no_grad():
        _, _, _, _, _, _, aux = build_forecast_inputs(model, tokenizer, batch, sample_fps=2, window_sec=1)
    a = aux[0]
    assert torch.allclose(a["future_latents"], a["copy_last_latents"][:, None].expand_as(a["future_latents"]))
    assert torch.allclose(a["future_boxes"], a["copy_last_boxes"][:, None].expand_as(a["future_boxes"]).float())
    assert a["teacher_valid"].all(), a["teacher_valid"]
    print("3. init == copy-last ok; teacher windows:", a["teacher_valid"].tolist())

    # 3b. the real path: a TRASER checkpoint (no forecaster) loaded as TRASER-F
    ckpt = os.path.join(root, "traser_ckpt")
    TRASER.save_pretrained(model, ckpt)  # saves forecaster too; drop it to mimic UWGZQ/TRASER
    from safetensors.torch import load, save_file
    with open(os.path.join(ckpt, "model.safetensors"), "rb") as f:  # not memory-mapped: we overwrite it
        sd = {k: v for k, v in load(f.read()).items() if "forecaster" not in k}
    save_file(sd, os.path.join(ckpt, "model.safetensors"), metadata={"format": "pt"})
    loaded = TRASERForecast.from_pretrained(ckpt, attn_implementation="eager")
    init_forecaster(loaded)
    assert all(torch.isfinite(p).all() for p in loaded.forecaster.parameters())
    with torch.no_grad():
        _, _, _, _, _, _, aux2 = build_forecast_inputs(loaded.eval(), tokenizer, batch, sample_fps=2, window_sec=1)
    assert torch.allclose(aux2[0]["future_latents"], aux2[0]["copy_last_latents"][:, None].expand_as(aux2[0]["future_latents"]))
    del loaded
    print("3b. TRASER checkpoint -> TRASER-F: forecaster initialized, starts at copy-last ok")

    # 4. two optimizer steps through the real trainer
    targs = ForecastTrainingArguments(
        output_dir=os.path.join(root, "out"), per_device_train_batch_size=1, max_steps=2, logging_steps=1,
        learning_rate=1e-3, forecaster_lr=1e-3, resampler_lr=1e-3, save_strategy="no", report_to=[],
        use_cpu=True, dataloader_num_workers=0, remove_unused_columns=False)
    trainer = ForecastTrainer(model=model, processing_class=tokenizer, args=targs, train_dataset=train_set,
                              data_collator=ForecastCollator(tokenizer.pad_token_id))
    model.train()
    loss = trainer.compute_loss(model, trainer._prepare_inputs(ForecastCollator(tokenizer.pad_token_id)([sample])))
    loss.backward()
    g = model.forecaster.out_proj.weight.grad
    assert torch.isfinite(loss) and g is not None and g.abs().sum() > 0
    print(f"4a. loss {loss.item():.3f}, forecaster grad norm {g.norm().item():.2e}, aux {dict(trainer._aux_sums)}")
    model.zero_grad()
    trainer._aux_sums.clear(); trainer._aux_count = 0
    result = trainer.train()
    print("4b. trainer.train() 2 steps ok, loss", round(result.training_loss, 3))

    # 5. evaluation path + metrics + baselines
    model.eval()
    eval_set = ForecastDataset(ann, processor, tokenizer, dargs, train=False)
    meter = ForecastMeter()
    s = eval_set[0]
    s["input_ids"], s["labels"] = eval_set.generation_prompt_ids(s), None
    b = history_only(ForecastCollator(tokenizer.pad_token_id)([s]), 2)
    with torch.no_grad():
        emb, pid, mask, rope, _, _, aux = build_forecast_inputs(model, tokenizer, b, sample_fps=2, window_sec=1,
                                                               with_teacher=False)
        gen = model.generate(inputs_embeds=emb, position_ids=pid, attention_mask=mask.long(), rope_deltas=rope,
                             max_new_tokens=8, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    meter.add(parse_relationships(text), s["meta"]["target"], 3, 2, aux[0]["future_boxes"], s["fut_boxes"],
              s["fut_box_valid"], aux[0]["vis_logits"])
    perfect = ForecastMeter()
    perfect.add(s["meta"]["target"], s["meta"]["target"], 3, 2, s["fut_boxes"], s["fut_boxes"], s["fut_box_valid"])
    assert perfect.summary()["R"] == 100.0 and perfect.summary()["mIoU@1s"] == 100.0
    print("5a. eval clips:", len(eval_set), "| random-model metrics:", meter.summary())

    items = json.load(open(ann))
    mask_data = json.load(open(items[0]["mask_json"]))
    ct = clip_targets(items[0], ClipSpec(0, 3, 3, 2), mask_data, dargs)
    markov = MarkovBaseline(min_count=1).fit(items)
    bm = ForecastMeter()
    bm.add(markov.predict(ct["observed"], ct["obj_names"], 3, 2), ct["target"], 3, 2,
           box_baseline(ct["boxes"], ct["box_valid"], 3, "velocity"), ct["boxes"][:, 3:], ct["box_valid"][:, 3:])
    cm = ForecastMeter()
    cm.add(copy_last_relations(ct["observed"], 3, 2), ct["target"], 3, 2)
    print("5b. markov+velocity:", bm.summary())
    print("    copy-last:", cm.summary())
    print("ALL OK")


if __name__ == "__main__":
    main()
