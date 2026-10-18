#!/bin/bash
# TRASER-F (scene graph forecasting) launcher. Works from any directory:
#   STAGE=A bash traser/scripts/train_forecast.sh      # forecaster only, TRASER frozen
#   STAGE=B bash traser/scripts/train_forecast.sh      # everything, starting from stage A
#
# Stage A trains only the forecaster (and its box head): the LM loss still reaches it through
# the frozen LLM, the latent-distillation targets come from the frozen resampler, so it is
# cheap and stable. Stage B unfreezes the LLM, the merger and the resamplers.
#
# Ablations (docs/FORECAST.md): MODE=prompt|copy|oracle, LAMBDA_LATENT=0, LAMBDA_BOX=0.
# Smoke run on one GPU:
#   NUM_GPUS=1 EXTRA_ARGS="--max_steps 2 --save_strategy no" STAGE=A bash traser/scripts/train_forecast.sh
set -euo pipefail
cd "$(dirname "$0")/.."

stage=${STAGE:-A}
mode=${MODE:-latent}
num_gpus=${NUM_GPUS:-8}
train_ann=${TRAIN_ANN:-data/psg4d_hoi_train.json}
lambda_latent=${LAMBDA_LATENT:-1.0}
lambda_box=${LAMBDA_BOX:-1.0}
report_to=${REPORT_TO:-none}
extra_args=${EXTRA_ARGS:-}
[ -f "${train_ann}" ] || { echo "train_forecast.sh: ${train_ann} not found - run data/prepare_psg4d_hoi.py first" >&2; exit 1; }

if [ "${stage}" = "A" ]; then
    init=${INIT:-UWGZQ/TRASER}
    output_dir=${OUTPUT_DIR:-./checkpoints/traser_f_${mode}_stageA}
    stage_args="--tune_mm_llm False --tune_mm_mlp False --tune_mm_resampler False --reinit_forecaster True
                --learning_rate 1e-4 --forecaster_lr 1e-4 --num_train_epochs ${EPOCHS:-20}"
else
    init=${INIT:-./checkpoints/traser_f_${mode}_stageA}
    output_dir=${OUTPUT_DIR:-./checkpoints/traser_f_${mode}_stageB}
    stage_args="--tune_mm_llm True --tune_mm_mlp True --tune_mm_resampler True --reinit_forecaster False
                --learning_rate 1e-5 --mm_projector_lr 2e-5 --resampler_lr 5e-5 --forecaster_lr 5e-5
                --num_train_epochs ${EPOCHS:-10}"
fi

args="
    --model_name_or_path ${init} \
    --train_annotation ${train_ann} \
    --forecast_mode ${mode} \
    --lambda_latent ${lambda_latent} \
    --lambda_box ${lambda_box} \
    --sample_fps 2 \
    --window_sec 1 \
    --history_min_sec 1 --history_max_sec 5 \
    --horizon_min_sec 1 --horizon_max_sec 5 \
    --object_filter related \
    --max_objects 20 \
    --object_names_in_prompt True \
    ${stage_args} \
    --tune_mm_vision False \
    --bf16 \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --coverage_thresh 0.5 \
    --time_reduce max \
    --run_name traser_f_${mode}_stage${stage} \
    --report_to ${report_to} \
    ${extra_args}"

deepspeed --num_gpus=${num_gpus} traser_train/forecast/train_forecast.py ${args} \
    --deepspeed ./scripts/zero3_offload.json
