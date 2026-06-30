#!/bin/bash

MODEL_PATH_LIST=(
    "ckpt/longlive"
    "ckpt/longlive"
)

OUTPUT_NAME_LIST=(
    "ours_longlive_longvideo_test"
    "ours_longlive_longvideo_test_2"
)

seed=(
    42 43
)


CONFIG_PATH="self_forcing/config/longlive_interactive_inference.yaml"

for ((i=0; i<${#MODEL_PATH_LIST[@]}; i++)); do

    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    OUTPUT_PATH=${OUTPUT_NAME_LIST[$i]}
    SEED=${seed[$i]}

    echo "========================================"
    echo "Running inference for: ${OUTPUT_PATH}"
    echo "Model path: ${MODEL_PATH}"
    echo "Seed: ${SEED}"
    echo "========================================"

    PROMPT_PATH="/path/to/TempAct/dataset/temporal_eval_simple_100.csv"

    torchrun --nproc_per_node=8 tools/inference_unified.py \
        --mode longlive \
        --prompt_path ${PROMPT_PATH} \
        --config_path ${CONFIG_PATH} \
        --lora_path ${MODEL_PATH} \
        --seed ${SEED} \
        --output_file "/path/to/output/longlive/simple_set_100/${OUTPUT_PATH}_36frame_gap12" \
        --sample_frames 36 \
        --gap_frame 12

    PROMPT_PATH="/path/to/TempAct/dataset/temporal_eval_combined_100.csv"

    torchrun --nproc_per_node=8 tools/inference_unified.py \
        --mode longlive \
        --prompt_path ${PROMPT_PATH} \
        --config_path ${CONFIG_PATH} \
        --lora_path ${MODEL_PATH} \
        --seed ${SEED} \
        --output_file "/path/to/output/longlive/hard_set_100/${OUTPUT_PATH}_60frame_gap12" \
        --sample_frames 60 \
        --gap_frame 12

done