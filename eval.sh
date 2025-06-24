#!/bin/bash

NOTE="ma_base" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="VLM"
RND_SEED=1
SCENARIO=$1 # 1 or 2

BATCHSIZE=4

##### Model settings (FIXED) #####
MODEL_NAME="./llava-v1.5-7b"
VERSION="v1"
VISION_TOWER="./clip-vit-large-patch14-336"
MODEL_TYPE="llama"
# MODEL_MAX_LEN=20000
BITS=16
###################################

# --master_port 29500
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED --per_gpu_train_batch_size $BATCHSIZE \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --scenario $SCENARIO \
    --output_dir "./results/test/" #> ./nohup/${NOTE}_sc${SCENARIO}.log 2>&1 &

# --eval_period $EVAL_PERIOD
#