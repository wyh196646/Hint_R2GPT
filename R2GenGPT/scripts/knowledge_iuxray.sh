#!/bin/bash

dataset="iu_xray"
annotation="/home/yuhaowang/project/report_generation/TRRG/R2GenGPT/merged_iuxray.json"
base_dir="/data2/yuhaowang/iu_xray/images"

version="v1_deep"
savepath="./save/$dataset/$version/knowledge"

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 24 \
    --val_batch_size 12 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 3 \
    --max_epochs 15 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt