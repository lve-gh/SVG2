#!/bin/bash
# TRASER training launcher. Works from any directory:
#   bash traser/scripts/train.sh
#
# The defaults reproduce the released checkpoint (all seven splits, 8 GPUs). Every
# annotation file named in DATASETS must exist under traser/data/ (see docs/DATA.md).
# Override with environment variables, e.g. a one-split smoke test on one GPU:
#   DATASETS=vidvrd NUM_GPUS=1 EXTRA_ARGS="--max_steps 2 --save_strategy no" bash traser/scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

datasets=${DATASETS:-svg2_sav,svg2_pvd,vipseg,vidor,vidvrd,lvvis,ovis}
num_gpus=${NUM_GPUS:-8}
output_dir=${OUTPUT_DIR:-./checkpoints/traser_qwen2.5vl_3b}
report_to=${REPORT_TO:-none}        # set REPORT_TO=wandb to log to Weights & Biases
extra_args=${EXTRA_ARGS:-}          # any further HF TrainingArguments, appended last

# Fail fast on a missing split instead of after the model has loaded.
for name in ${datasets//,/ }; do
    ann="data/${name%%%*}.json"
    [ -f "${ann}" ] || { echo "train.sh: ${ann} not found - convert this split first (docs/DATA.md) or narrow DATASETS" >&2; exit 1; }
done

deepspeed_config=./scripts/zero3_offload.json
llm=Qwen/Qwen2.5-VL-3B-Instruct
entry_file=traser_train/train/train_qwen.py
run_name="traser_qwen2.5vl_3b"

# Training hyperparameters
lr=2e-5
resampler_lr=1e-4
mm_projector_lr=5e-5
batch_size=1
grad_accum_steps=2

args="
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --tune_mm_resampler True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 602112 \
    --min_pixels 50176 \
    --eval_strategy no \
    --save_strategy steps \
    --remove_unused_columns False \
    --save_steps 1000 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --resampler_lr ${resampler_lr} \
    --save_total_limit 45 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 128000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to ${report_to} \
    --coverage_thresh 0.5 \
    --time_reduce max \
    --temporal_window_length 4 \
    --resampler_depth 3 \
    --temporal_resampler_n_latents 32 \
    --object_resampler_n_latents 32 \
    --object_resampler True \
    ${extra_args}"

deepspeed --num_gpus=${num_gpus} \
         ${entry_file} \
         ${args} \
         --deepspeed "${deepspeed_config}"
