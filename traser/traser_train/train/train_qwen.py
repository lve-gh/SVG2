# Adopted from https://github.com/QwenLM/Qwen2.5-VL and https://github.com/lm-sys/FastChat.
# Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import pathlib
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import traser_train.train.trainer  # noqa: F401  (installs the optimizer / parameter-printing patches)
from traser_train.data.data_qwen import make_supervised_data_module
from traser_train.train.argument import DataArguments, ModelArguments, TrainingArguments
from traser_train.train.modeling_traser import TRASER
from traser_train.train.trainer_insert import TraserTrainer

# Special tokens wrapping each object-trajectory block.
OBJ_TRAJ_TOKENS = ["<obj_traj_start>", "<obj_traj_end>"]

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def add_obj_traj_tokens(model, tokenizer, data_args):
    """Add the <obj_traj_start> / <obj_traj_end> special tokens and grow the embeddings."""
    # Pass the union of the existing and the new special tokens, which behaves the same
    # across transformers 4.x and 5.x (the replace_* keyword was renamed in 5.x).
    current_specials = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    new_specials = list(dict.fromkeys(current_specials + OBJ_TRAJ_TOKENS))
    num_added = 0

    if new_specials != current_specials:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": new_specials}
        )
    if getattr(data_args, "processor", None) is not None and hasattr(data_args.processor, "tokenizer"):
        data_args.processor.tokenizer = tokenizer

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

        # Initialize the new embedding rows from the standard deviation of the old ones.
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            old_n = emb.shape[0] - num_added
            std = emb[:old_n].std().item()
            emb[old_n:].normal_(mean=0.0, std=std)


def get_visual(model):
    # transformers >= 5 removed the top-level convenience accessors.
    return model.visual if hasattr(model, "visual") else model.model.visual


def get_language_model(model):
    return model.language_model if hasattr(model, "language_model") else model.model.language_model


def set_model(model_args, training_args, model):
    if model_args.tune_mm_vision:
        for n, p in get_visual(model).named_parameters():
            p.requires_grad = True
    else:
        for n, p in get_visual(model).named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in get_visual(model).merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in get_visual(model).merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in get_language_model(model).named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in get_language_model(model).named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    resampler_modules = [model.perceiver_resampler]
    if hasattr(model, "second_perceiver_resampler"):
        resampler_modules.append(model.second_perceiver_resampler)
    for module in resampler_modules:
        for p in module.parameters():
            p.requires_grad = training_args.tune_mm_resampler

    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.seed = 42
    training_args.data_seed = 42

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = TRASER.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        use_resampler=training_args.use_resampler,
        object_resampler=training_args.object_resampler,
        resampler_depth=training_args.resampler_depth,
        temporal_resampler_n_latents=training_args.temporal_resampler_n_latents,
        object_resampler_n_latents=training_args.object_resampler_n_latents,
    )
    data_args.processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    add_obj_traj_tokens(model, tokenizer, data_args)
    training_args.obj_traj_start_id = tokenizer.convert_tokens_to_ids([OBJ_TRAJ_TOKENS[0]])[0]
    training_args.obj_traj_end_id = tokenizer.convert_tokens_to_ids([OBJ_TRAJ_TOKENS[1]])[0]
    model.config.obj_traj_start_id = training_args.obj_traj_start_id
    model.config.obj_traj_end_id = training_args.obj_traj_end_id
    model.config.vocab_size = len(tokenizer)

    set_model(model_args, training_args, model)
    if local_rank in (0, -1):
        get_visual(model).print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = TraserTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
