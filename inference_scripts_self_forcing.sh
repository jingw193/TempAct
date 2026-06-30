#!/bin/bash

MODEL_PATH_LIST=(
    "ckpt/self_forcing"
)

OUTPUT_NAME_LIST=(
    "self_forcing_run"
)

BASE_MODEL_PATH="/path/to/hf_cache/models--gdhe17--Self-Forcing/snapshots/2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/self_forcing_dmd.pt"

CONFIG_PATH="self_forcing/config/self_forcing_dmd.yaml"

for ((i=0; i<${#MODEL_PATH_LIST[@]}; i++)); do

    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    OUTPUT_PATH=${OUTPUT_NAME_LIST[$i]}

    echo "========================================"
    echo "Running inference for: ${OUTPUT_PATH}"
    echo "Model path: ${MODEL_PATH}"
    echo "========================================"

    # simple100
    PROMPT_PATH="/path/to/TempAct/dataset/temporal_eval_simple_100.csv"

    torchrun --nproc_per_node=8 tools/inference_unified.py \
        --mode self_forcing \
        --prompt_path ${PROMPT_PATH} \
        --config_path ${CONFIG_PATH} \
        --model_path ${BASE_MODEL_PATH} \
        --lora_path ${MODEL_PATH} \
        --output_file "/path/to/output/self_forcing/simple_set_100/${OUTPUT_PATH}_36frame_gap12" \
        --sample_frames 36 \
        --gap_frame 12

    # middle100
    PROMPT_PATH="/path/to/TempAct/dataset/temporal_eval_combined_100.csv"

    torchrun --nproc_per_node=8 tools/inference_unified.py \
        --mode self_forcing \
        --prompt_path ${PROMPT_PATH} \
        --config_path ${CONFIG_PATH} \
        --model_path ${BASE_MODEL_PATH} \
        --lora_path ${MODEL_PATH} \
        --output_file "/path/to/output/self_forcing/hard_set_100/${OUTPUT_PATH}_60frame_gap12" \
        --sample_frames 60 \
        --gap_frame 12

done