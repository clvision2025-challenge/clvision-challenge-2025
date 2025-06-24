#!/bin/bash
NOTE="baseline" # Name of the experiment
MODE="VLM"
RND_SEED=4
SCENARIO=$1 # 1 or 2

NUM_ITER=0.5
BATCHSIZE=2
GAS=2 # gradient accumulation steps
LR=1e-5
MM_PROJECTOR_LR=5e-5
OPT_NAME="adamw_torch"
SCHED_NAME="constant"
WARMUP_RATIO=0.03

##### Model settings (FIXED) #####
MODEL_NAME="./llava-v1.5-7b"
VERSION="v1"
VISION_TOWER="./clip-vit-large-patch14-336"
MODEL_TYPE="llama"
BITS=16
###################################


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 27006 \
--include localhost:0 \
main.py \
--deepspeed ./deepspeed_script/zero2.json \
--gradient_checkpointing True \
--model_name_or_path $MODEL_NAME \
--model_name_for_dataarg $MODEL_NAME \
--model_type $MODEL_TYPE \
--vision_tower $VISION_TOWER \
--version $VERSION \
--bits $BITS \
--bf16 True \
--tf32 True \
--mode $MODE --dataloader_num_workers 2 \
--weight_decay 0. \
--warmup_ratio $WARMUP_RATIO \
--num_train_epochs 1 \
--gradient_accumulation_steps $GAS \
--learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
--mm_projector_lr $MM_PROJECTOR_LR \
--optim $OPT_NAME \
--lr_scheduler_type $SCHED_NAME \
--evaluation_strategy "no" \
--save_strategy "no" \
--logging_steps 2 \
--num_iter $NUM_ITER \
--note $NOTE \
--scenario $SCENARIO \
--output_dir "./results/test/" # > ./nohup/${NOTE}.log 2>&1 &
